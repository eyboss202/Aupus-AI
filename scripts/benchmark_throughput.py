"""Measure real training throughput (tokens/sec) for each candidate model preset.

Why this exists: the token budget for Aupus 0.1 is fixed by Chinchilla-style
scaling (~20 tokens per parameter), so the only unknown in "how long will
training take?" is how many tokens per second this machine can push through a
full training step (forward + backward + AdamW update). We measure that instead
of guessing.

NOTE (Phase 0 only): this file contains a *compact, self-contained* Llama-style
model with the same shapes and FLOP profile as the planned architecture
(RMSNorm, RoPE, GQA via SDPA, SwiGLU, tied embeddings). It exists so we can
benchmark before Milestone 4. Once `aupus.model` exists, this script will import
the real model instead and this copy will be deleted.

Usage:
    python scripts/benchmark_throughput.py --presets tiny small --context 1024 --micro-batch 1
"""

from __future__ import annotations

import argparse
import json
import math
import resource
import time
from dataclasses import asdict

import torch
import torch.nn.functional as F
from torch import nn

from aupus.model.config import PRESET_DIR, ModelConfig, flops_per_token, param_count

PRESETS = sorted(p.stem for p in PRESET_DIR.glob("*.yaml"))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rope_cache(seq_len: int, head_dim: int, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(torch.arange(seq_len).float(), inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class Block(nn.Module):
    def __init__(self, p: ModelConfig) -> None:
        super().__init__()
        self.p = p
        hd = p.d_model // p.n_heads
        self.norm1, self.norm2 = RMSNorm(p.d_model), RMSNorm(p.d_model)
        self.wq = nn.Linear(p.d_model, p.n_heads * hd, bias=False)
        self.wkv = nn.Linear(p.d_model, 2 * p.n_kv_heads * hd, bias=False)
        self.wo = nn.Linear(p.n_heads * hd, p.d_model, bias=False)
        self.w_gate_up = nn.Linear(p.d_model, 2 * p.ffn_hidden, bias=False)
        self.w_down = nn.Linear(p.ffn_hidden, p.d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        p, hd = self.p, self.p.d_model // self.p.n_heads
        h = self.norm1(x)
        q = self.wq(h).view(b, t, p.n_heads, hd).transpose(1, 2)
        k, v = self.wkv(h).view(b, t, 2, p.n_kv_heads, hd).unbind(2)
        k, v = k.transpose(1, 2), v.transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        x = x + self.wo(attn.transpose(1, 2).reshape(b, t, -1))
        gate, up = self.w_gate_up(self.norm2(x)).chunk(2, dim=-1)
        return x + self.w_down(F.silu(gate) * up)


class BenchModel(nn.Module):
    def __init__(self, p: ModelConfig, context: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(p.vocab_size, p.d_model)
        self.blocks = nn.ModuleList(Block(p) for _ in range(p.n_layers))
        self.norm = RMSNorm(p.d_model)
        cos, sin = rope_cache(context, p.d_model // p.n_heads)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        # GPT-2-style init: N(0, 0.02). PyTorch's default N(0, 1) embedding init would
        # produce huge logits through the tied LM head (initial loss >> ln(vocab)).
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        t = idx.shape[1]
        x = self.embed(idx)
        for blk in self.blocks:
            x = blk(x, self.cos[:t], self.sin[:t])
        logits = self.norm(x) @ self.embed.weight.T  # weight tying
        return F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1))


def bench(
    name: str,
    p: ModelConfig,
    context: int,
    micro_batch: int,
    steps: int,
    warmup: int,
    dtype: str,
    compile_: bool,
) -> dict:
    torch.manual_seed(0)
    model = BenchModel(p, context)
    if compile_:
        model = torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=False)
    x = torch.randint(0, p.vocab_size, (micro_batch, context))
    y = torch.randint(0, p.vocab_size, (micro_batch, context))
    use_bf16 = dtype == "bf16"

    def step() -> float:
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=use_bf16):
            loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        return loss.item()

    t0 = time.perf_counter()
    for _ in range(warmup):
        step()
    warm_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(steps):
        loss = step()
    elapsed = time.perf_counter() - t0
    tok_s = steps * micro_batch * context / elapsed
    flops = flops_per_token(p, context)["train"]
    counts = param_count(p)
    total, non_embed = counts["total"], counts["non_embedding"]
    return {
        "preset": name,
        **asdict(p),
        "context_benchmarked": context,
        "params_total_M": round(total / 1e6, 1),
        "params_non_embedding_M": round(non_embed / 1e6, 1),
        "micro_batch": micro_batch,
        "dtype": dtype,
        "compiled": compile_,
        "tokens_per_sec": round(tok_s, 1),
        "achieved_gflops": round(tok_s * flops / 1e9, 1),
        "sec_per_step": round(elapsed / steps, 2),
        "warmup_sec": round(warm_s, 1),
        "peak_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 2),
        "final_loss": round(loss, 3),
        "sanity_initial_loss_expected": round(math.log(p.vocab_size), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--presets", nargs="+", default=PRESETS, choices=PRESETS)
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--compile", action="store_true", help="wrap the model in torch.compile")
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads (default: torch's choice)")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cpu":
        raise SystemExit(
            "This Phase-0 benchmark only implements the CPU path; extend it when a GPU is available."
        )
    print(f"# torch {torch.__version__} | device={device} | threads={torch.get_num_threads()}")
    for name in args.presets:
        r = bench(
            name,
            ModelConfig.preset(name),
            args.context,
            args.micro_batch,
            args.steps,
            args.warmup,
            args.dtype,
            args.compile,
        )
        print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
