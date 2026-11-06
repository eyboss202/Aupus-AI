"""Training: pretraining loop, AdamW + LR schedule, full-state checkpoints (SFT arrives in Milestone 10)."""

from aupus.train.config import OptimConfig, TrainConfig, resolve_model_config
from aupus.train.optim import lr_at, make_optimizer, param_groups
from aupus.train.pretrain import Trainer, compile_unsupported_reason, resolve_device, resolve_precision

__all__ = [
    "OptimConfig",
    "TrainConfig",
    "Trainer",
    "compile_unsupported_reason",
    "lr_at",
    "make_optimizer",
    "param_groups",
    "resolve_device",
    "resolve_model_config",
    "resolve_precision",
]
