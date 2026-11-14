"""The Vulpes decoder-only transformer.

    token ids -> embedding -> n_layers x Block -> RMSNorm -> LM head -> logits

With ``tie_embeddings`` (the default) the LM head reuses the embedding matrix:
the vector used to *read* token ``i`` is also used to *score* token ``i`` as the
next token. That saves ``vocab_size * d_model`` parameters (24.6M of the 125M
in ``small``) and works as well as a separate head at this scale.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from vulpes.model.config import ModelConfig
from vulpes.model.layers import Block, KVCache, RMSNorm, RotaryEmbedding

#: Target value that is excluded from the loss: padding, and in SFT every
#: token that isn't part of an assistant reply.
IGNORE_INDEX = -100


class Output(NamedTuple):
    logits: torch.Tensor  # (batch, seq, vocab)
    loss: torch.Tensor | None  # mean cross-entropy over non-ignored targets, if targets were given


class Transformer(nn.Module):
    """Llama-style decoder: RMSNorm, RoPE, GQA, SwiGLU, no biases, optional tied head.

    Attributes:
        gradient_checkpointing: if True (and training), each block's activations
            are recomputed in the backward pass instead of stored. That costs
            ~33% more compute but cuts activation memory roughly
            ``n_layers``-fold. It's what lets ``small`` train on a 4 GiB GPU.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.gradient_checkpointing = False
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = None if cfg.tie_embeddings else nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.context_length, cfg.rope_theta)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialise weights the GPT-2 way.

        * every matmul weight and the embedding: N(0, init_std), with init_std = 0.02;
        * the two projections that write *into* the residual stream (attention
          ``wo``, MLP ``w_down``) get std / sqrt(2 * n_layers). The residual
          stream sums 2 * n_layers such outputs, so this keeps its variance
          roughly independent of depth at init;
        * RMSNorm weights: 1.

        With tied embeddings the initial logits are tiny, so the initial loss is
        close to ln(vocab_size), the loss of a uniform guess. ``test_model.py``
        checks this.
        """
        std = self.cfg.init_std
        residual_std = std / math.sqrt(2 * self.cfg.n_layers)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                is_residual_out = name.endswith(("attn.wo", "mlp.w_down"))
                nn.init.normal_(module.weight, mean=0.0, std=residual_std if is_residual_out else std)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)

    def num_params(self) -> int:
        """Unique parameters (a tied embedding is counted once)."""
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
        start_pos: int = 0,
    ) -> Output:
        """Run the model.

        Args:
            idx: token ids, (batch, seq).
            targets: next-token ids, (batch, seq), or None. Positions equal to
                ``IGNORE_INDEX`` don't count towards the loss.
            kv_cache: optional cache for generation (see ``generate.py``).
            start_pos: position of ``idx[:, 0]`` (non-zero only when continuing
                from a filled ``kv_cache``).
        """
        _, t = idx.shape
        cos, sin = self.rope(start_pos, t)
        x = self.embed(idx)
        use_checkpointing = self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        for block in self.blocks:
            if use_checkpointing and kv_cache is None:
                x = checkpoint(block, x, cos, sin, use_reentrant=False)
            else:
                x = block(x, cos, sin, kv_cache, start_pos)
        x = self.norm(x)
        logits = F.linear(x, self.embed.weight) if self.lm_head is None else self.lm_head(x)

        loss = None
        if targets is not None:
            # Loss in fp32: log-softmax over 32k classes in fp16 loses too much precision.
            loss = F.cross_entropy(
                logits.float().view(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=IGNORE_INDEX
            )
        return Output(logits, loss)
