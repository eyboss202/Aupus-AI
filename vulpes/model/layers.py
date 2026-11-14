"""Building blocks of the Vulpes transformer: RMSNorm, RoPE, attention (GQA + KV cache), SwiGLU.

Every block is pre-norm: ``x + f(norm(x))``. With the norm *inside* the
residual branch, the residual stream is an untouched sum of every block's
output, so gradients reach early layers easily and training is stable without
learning-rate tricks. That's why GPT-2 and later models dropped post-norm.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from vulpes.model.config import ModelConfig


class RMSNorm(nn.Module):
    """Root-mean-square norm (Zhang & Sennrich 2019): ``x / rms(x) * weight``.

    LayerNorm subtracts the mean and divides by the std. RMSNorm skips the mean
    subtraction (it turned out not to matter) and has no bias. It's cheaper and
    works just as well.

    The statistics are computed in fp32 even under fp16/bf16 autocast. Squaring
    activations in fp16 overflows above ~256, and the GTX 1650-class GPUs we
    might train on only have fp16.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return normed.to(x.dtype) * self.weight.to(x.dtype)


class RotaryEmbedding(nn.Module):
    """Rotary position embedding tables (Su et al. 2021, "RoFormer").

    RoPE encodes position by *rotating* each pair of query/key dimensions by an
    angle proportional to the position: ``pos * theta^(-2i/d)`` for pair ``i``.
    The dot product of a query rotated by angle ``a`` and a key rotated by
    angle ``b`` depends only on ``a - b``, so attention scores see *relative*
    positions. No position parameters are learned.

    We use the "rotate half" layout: pair ``i`` is dims ``(i, i + d/2)``, not
    adjacent dims ``(2i, 2i+1)``. Both are equivalent once trained; rotate-half
    needs no strided indexing, so it's faster.
    """

    def __init__(self, head_dim: int, max_len: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        angles = torch.outer(torch.arange(max_len, dtype=torch.float32), inv_freq)  # (max_len, head_dim/2)
        angles = torch.cat((angles, angles), dim=-1)  # (max_len, head_dim), rotate-half layout
        # Not persistent: derived from the config, so checkpoints don't need to store them.
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, start: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        """cos/sin tables for positions ``start .. start+length-1``, shape (length, head_dim)."""
        if start + length > self.cos.shape[0]:
            raise ValueError(f"position {start + length - 1} exceeds the RoPE table ({self.cos.shape[0]})")
        return self.cos[start : start + length], self.sin[start : start + length]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate ``x`` of shape (batch, heads, seq, head_dim) by the given tables."""
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)  # (x1, x2) -> (-x2, x1)
    return x * cos.to(x.dtype) + rotated * sin.to(x.dtype)


class KVCache:
    """Pre-allocated key/value cache for fast autoregressive generation.

    Without a cache, generating token ``n`` recomputes keys and values for all
    ``n - 1`` previous tokens. With one, each step only computes the new
    token's K/V and appends them. The cache stores the ``n_kv_heads`` heads
    *before* they're shared across query heads, so GQA's 3x saving (12 query
    heads, 4 KV heads) applies to memory here.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        batch_size: int,
        max_len: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.max_len = max_len or cfg.context_length
        shape = (batch_size, cfg.n_kv_heads, self.max_len, cfg.head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.n_layers)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.n_layers)]

    def update(
        self, layer: int, start: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Store K/V for positions ``start ..`` and return K/V for all positions so far."""
        end = start + k.shape[2]
        if end > self.max_len:
            raise ValueError(f"KV cache full: position {end - 1} >= max_len {self.max_len}")
        self.k[layer][:, :, start:end] = k
        self.v[layer][:, :, start:end] = v
        return self.k[layer][:, :, :end], self.v[layer][:, :, :end]


class Attention(nn.Module):
    """Causal self-attention with grouped-query attention (GQA, Ainslie et al. 2023).

    Each of the ``n_kv_heads`` key/value heads is shared by
    ``n_heads // n_kv_heads`` query heads. That cuts K/V parameters and the KV
    cache by that factor at almost no quality cost.

    Q, K and V come from one fused matmul (one big kernel beats three small
    ones). Before attention, shared K/V heads are expanded to match the query
    heads with ``expand`` (a view, no copy, until ``reshape``). We don't use
    SDPA's ``enable_gqa`` flag because the memory-efficient kernel, the fast
    path on pre-Ampere GPUs like Turing, doesn't support it.
    """

    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.n_heads, self.n_kv_heads, self.head_dim = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.wqkv = nn.Linear(cfg.d_model, (cfg.n_heads + 2 * cfg.n_kv_heads) * cfg.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: KVCache | None = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        b, t, _ = x.shape
        hd, nh, nkv = self.head_dim, self.n_heads, self.n_kv_heads
        q, k, v = self.wqkv(x).split((nh * hd, nkv * hd, nkv * hd), dim=-1)
        q = q.view(b, t, nh, hd).transpose(1, 2)  # (b, heads, t, head_dim)
        k = k.view(b, t, nkv, hd).transpose(1, 2)
        v = v.view(b, t, nkv, hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        if kv_cache is not None:
            k, v = kv_cache.update(self.layer_idx, start_pos, k, v)
        if nkv != nh:
            rep = nh // nkv
            tk = k.shape[2]
            k = k[:, :, None].expand(b, nkv, rep, tk, hd).reshape(b, nh, tk, hd)
            v = v[:, :, None].expand(b, nkv, rep, tk, hd).reshape(b, nh, tk, hd)

        if start_pos == 0:
            # Prefill / training: plain causal mask (the fused kernels handle it internally).
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif t == 1:
            # One new token attending to everything cached: no mask needed.
            out = F.scaled_dot_product_attention(q, k, v)
        else:
            # A chunk appended after cached tokens. SDPA's is_causal aligns the mask
            # top-left, which is wrong when queries start at start_pos, so build it.
            q_pos = torch.arange(start_pos, start_pos + t, device=x.device)
            mask = torch.arange(k.shape[2], device=x.device)[None, :] <= q_pos[:, None]
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.wo(out.transpose(1, 2).reshape(b, t, nh * hd))


class SwiGLU(nn.Module):
    """Gated MLP (Shazeer 2020): ``down(silu(gate(x)) * up(x))``.

    The gate lets each hidden unit scale its own output. With matched parameter
    counts this beats a plain GELU MLP. Gate and up projections are fused into
    one matmul.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.w_gate_up = nn.Linear(cfg.d_model, 2 * cfg.ffn_hidden, bias=False)
        self.w_down = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w_gate_up(x).chunk(2, dim=-1)
        return self.w_down(F.silu(gate) * up)


class Block(nn.Module):
    """One transformer block: ``x + attn(norm(x))``, then ``x + mlp(norm(x))``."""

    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg, layer_idx)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: KVCache | None = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin, kv_cache, start_pos)
        return x + self.mlp(self.mlp_norm(x))
