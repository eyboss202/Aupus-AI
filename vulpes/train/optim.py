"""AdamW with selective weight decay, and the learning-rate schedules (cosine and WSD).

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

**Schedules.** Both start with a linear warmup from ~0 to ``lr`` over
``warmup_steps`` (Adam's second-moment estimates are noisy for the first few
hundred steps, and large early updates can push the model into a bad region).

* ``cosine`` (Vulpes 0.1): a half cosine from ``lr`` down to ``min_lr`` at
  ``max_steps``. Every step's LR depends on ``max_steps``, so the run's length
  has to be fixed at the start.
* ``wsd``, warmup-stable-decay (Vulpes Flos 1.0): ``lr`` held flat, then a
  decay to ``min_lr`` over the last ``decay_fraction`` of ``max_steps``, shaped
  ``1 - sqrt(progress)`` (Hägele et al. 2024, "Scaling Laws and Compute-Optimal
  Training Beyond Fixed Training Durations": it matches cosine at the same
  length, and that shape beat a linear decay there). On the plateau the LR does
  not depend on ``max_steps``, so a run paid for in weekly quota can be
  lengthened or shortened mid-way (``--resume ... max_steps=N``) for as long as
  the decay hasn't started. ``max_steps_to_decay_from`` gives the ``max_steps``
  that starts the decay at a chosen step.

The LR is a pure function of the step number, so resuming only needs the step:
no scheduler state to save (decision 21).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from vulpes.train.config import OptimConfig


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


def decay_start(cfg: OptimConfig, max_steps: int) -> int:
    """First step of the WSD decay (``max_steps`` for cosine, which has no plateau)."""
    if cfg.schedule != "wsd":
        return cfg.warmup_steps
    return max(cfg.warmup_steps, max_steps - max(1, round(cfg.decay_fraction * max_steps)))


def max_steps_to_decay_from(step: int, cfg: OptimConfig) -> int:
    """The smallest ``max_steps`` whose WSD decay starts at ``step`` or later.

    For deciding mid-run (docs/PLAN-FLOS.md, D2): to begin the decay now, at step S,
    resume with ``max_steps=max_steps_to_decay_from(S, cfg)``.
    """
    if cfg.schedule != "wsd":
        raise ValueError("only the wsd schedule has a decay that can be started mid-run")
    if cfg.decay_fraction == 1 and step > cfg.warmup_steps:
        raise ValueError("decay_fraction 1 decays from the end of warmup: there is no plateau to end")
    m = step  # decay_start never decreases as max_steps grows, so the first hit is the smallest
    while decay_start(cfg, m) < step:
        m += 1
    return m


def lr_at(step: int, cfg: OptimConfig, max_steps: int) -> float:
    """Learning rate for optimizer step ``step`` (0-based)."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if max_steps <= cfg.warmup_steps:
        return cfg.lr
    if cfg.schedule == "wsd":
        start = decay_start(cfg, max_steps)
        if step < start:
            return cfg.lr
        progress = min(1.0, (step - start) / (max_steps - start))
        return cfg.min_lr + (1.0 - math.sqrt(progress)) * (cfg.lr - cfg.min_lr)
    progress = min(1.0, (step - cfg.warmup_steps) / (max_steps - cfg.warmup_steps))
    return cfg.min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (cfg.lr - cfg.min_lr)


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr
