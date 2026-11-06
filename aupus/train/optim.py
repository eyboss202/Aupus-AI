"""AdamW with selective weight decay, and the warmup + cosine learning-rate schedule.

**Which parameters get weight decay.** Weight decay pulls weights towards 0,
which regularises the big matmul matrices. It's wrong for two kinds of
parameter:

* RMSNorm gains are initialised to 1 and set the scale of each channel. Decay
  would drag them towards 0, which only fights the optimizer;
* the token embedding: a row only gets gradient when its token appears, but
  decay shrinks *every* row at *every* step, so rare tokens' embeddings would
  decay away. (With tied weights this matrix is also the LM head; the usual
  Llama/GPT-NeoX recipe still leaves it undecayed.)

So decay goes to the 2-D weights of ``nn.Linear`` layers only.

**Schedule.** Linear warmup from ~0 to ``lr`` over ``warmup_steps`` (Adam's
second-moment estimates are noisy for the first few hundred steps, and large
early updates can push the model into a bad region), then a half cosine from
``lr`` down to ``min_lr`` at ``max_steps``. The LR is a pure function of the
step number, so resuming only needs the step: no scheduler state to save.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from aupus.train.config import OptimConfig


def param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Split parameters into a decayed group (Linear weights) and a non-decayed group (the rest)."""
    decay_ids = {id(m.weight) for m in model.modules() if isinstance(m, nn.Linear)}
    decay, no_decay = [], []
    for p in model.parameters():  # unique parameters: a tied weight appears once
        if p.requires_grad:
            (decay if id(p) in decay_ids else no_decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def make_optimizer(model: nn.Module, cfg: OptimConfig, device: torch.device) -> torch.optim.AdamW:
    """AdamW; on CUDA the fused kernel updates all parameters in a few launches instead of one per tensor."""
    return torch.optim.AdamW(
        param_groups(model, cfg.weight_decay),
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
        fused=device.type == "cuda",
    )


def lr_at(step: int, cfg: OptimConfig, max_steps: int) -> float:
    """Learning rate for optimizer step ``step`` (0-based)."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if max_steps <= cfg.warmup_steps:
        return cfg.lr
    progress = min(1.0, (step - cfg.warmup_steps) / (max_steps - cfg.warmup_steps))
    return cfg.min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (cfg.lr - cfg.min_lr)


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr
