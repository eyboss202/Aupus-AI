"""Training: pretraining loop, supervised fine-tuning, AdamW + LR schedule, full-state checkpoints."""

from aupus.train.config import OptimConfig, TrainConfig, resolve_model_config
from aupus.train.optim import lr_at, make_optimizer, param_groups
from aupus.train.pretrain import Trainer, compile_unsupported_reason, resolve_device, resolve_precision
from aupus.train.sft import SFTConfig, SFTTrainer

__all__ = [
    "OptimConfig",
    "SFTConfig",
    "SFTTrainer",
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
