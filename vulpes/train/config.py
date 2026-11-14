"""Pretraining run configuration (strict YAML, like every other config in Vulpes).

One YAML file in ``configs/train/`` describes a whole run: which model, which
data, how big a batch, how long, and how often to evaluate, sample and save.
The resolved config (after command-line overrides) is written into the run
directory as ``config.yaml``, so every checkpoint can be traced back to the
exact settings that produced it.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from vulpes.config import ConfigError, from_dict, load_yaml
from vulpes.model.config import ModelConfig

PRECISIONS = ("auto", "fp32", "bf16", "fp16")
DEVICES = ("auto", "cpu", "cuda")
SCHEDULES = ("cosine", "wsd")
T = TypeVar("T")


@dataclass(frozen=True)
class OptimConfig:
    """AdamW + linear warmup + a cosine or warmup-stable-decay schedule (see ``optim.py``).

    Defaults are the GPT-3-small / Llama recipe from docs/PLAN.md §4 (cosine, Vulpes 0.1).
    ``schedule: wsd`` holds ``lr`` after warmup and decays to ``min_lr`` over the last
    ``decay_fraction`` of ``max_steps`` (Vulpes Flos 1.0, docs/PLAN-FLOS.md).
    """

    lr: float = 6e-4  # peak learning rate, reached at the end of warmup
    min_lr: float = 6e-5  # final learning rate at ``max_steps``
    warmup_steps: int = 190
    weight_decay: float = 0.1  # matmul weights only; never norms or embeddings
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0  # max global grad norm; 0 disables clipping
    schedule: str = "cosine"  # "cosine" or "wsd"
    decay_fraction: float = 0.15  # wsd only: the share of max_steps spent decaying

    def __post_init__(self) -> None:
        if not 0 <= self.min_lr <= self.lr:
            raise ConfigError(f"need 0 <= min_lr <= lr, got min_lr={self.min_lr}, lr={self.lr}")
        if self.warmup_steps < 0 or self.grad_clip < 0 or self.weight_decay < 0:
            raise ConfigError("warmup_steps, grad_clip and weight_decay must be >= 0")
        if self.schedule not in SCHEDULES:
            raise ConfigError(f"schedule must be one of {SCHEDULES}, got {self.schedule!r}")
        if not 0 < self.decay_fraction <= 1:
            raise ConfigError(f"decay_fraction must be in (0, 1], got {self.decay_fraction}")


@dataclass(frozen=True)
class TrainConfig:
    """Everything about one pretraining run.

    Attributes:
        run_name: the run directory is ``<out_dir>/<run_name>``.
        model: a model preset name (``configs/model/<name>.yaml``) or a path to a model YAML.
        data_dir: prepared dataset directory (``meta.json`` + shards).
        tokenizer: tokenizer JSON; must be the one the shards were made with (checked by hash).
        seq_len: training context length. The model is built with
            ``context_length = seq_len``, so it never sees positions it wasn't trained on.
            ``None`` keeps the preset's context length.
        micro_batch: sequences per forward/backward pass (bounded by memory).
        grad_accum: micro-batches summed into one optimizer step. Effective batch is
            ``micro_batch * grad_accum * seq_len`` tokens per step. With several GPUs
            (``torchrun``, ``distributed.py``) it is still the step's total: each of W
            processes does ``grad_accum / W`` of them, so it must divide evenly.
        max_steps: optimizer steps; the learning-rate schedule ends here. With
            ``optim.schedule: wsd`` it can still be changed on ``--resume`` until the decay
            has started (the plateau doesn't depend on it).
        device: ``auto`` (CUDA if available), ``cpu`` or ``cuda``.
        precision: ``auto`` picks bf16 on GPUs with bf16 tensor cores (Ampere+) and
            fp32 everywhere else; fp16 must be asked for explicitly (docs/PLAN.md decision 13).
        compile: use ``torch.compile`` when this machine supports it (falls back to eager otherwise).
        grad_checkpointing: recompute activations in backward to save memory.
        cuda_memory_limit_gib: the most GPU memory this run may reserve (``None`` = no
            limit). Past it PyTorch frees its cache, then fails with a clear out-of-memory
            error. Without a limit, Windows silently moves the excess into system RAM and
            training gets several times slower (docs/PLAN.md §18–§19).
        eval_every / eval_batches: validation loss every N steps on the first
            ``eval_batches`` micro-batches of the val split (the same tokens every time).
        save_every / keep_last: checkpoint every N steps, keep the newest ``keep_last``
            (plus ``best.pt``, the lowest validation loss so far).
        log_every: print + log training metrics every N steps.
        sample_every / sample_prompts / sample_tokens: generate text from fixed prompts
            every N steps (0 = never), to watch quality improve.
        seed: seeds model init, data order and sampling.
    """

    run_name: str
    model: str
    data_dir: str
    tokenizer: str = "data/tokenizer/vulpes-bpe-32k.json"
    out_dir: str = "runs"
    seq_len: int | None = None
    micro_batch: int = 8
    grad_accum: int = 1
    max_steps: int = 1000
    optim: OptimConfig = field(default_factory=OptimConfig)
    device: str = "auto"
    precision: str = "auto"
    compile: bool = True
    grad_checkpointing: bool = False
    cuda_memory_limit_gib: float | None = None
    eval_every: int = 250
    eval_batches: int = 20
    save_every: int = 500
    keep_last: int = 3
    log_every: int = 10
    sample_every: int = 0
    sample_prompts: list[str] = field(default_factory=list)
    sample_tokens: int = 64
    seed: int = 1337

    def __post_init__(self) -> None:
        if self.device not in DEVICES:
            raise ConfigError(f"device must be one of {DEVICES}, got {self.device!r}")
        if self.precision not in PRECISIONS:
            raise ConfigError(f"precision must be one of {PRECISIONS}, got {self.precision!r}")
        for name in ("micro_batch", "grad_accum", "max_steps", "log_every", "keep_last", "sample_tokens"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"{name} must be positive, got {getattr(self, name)}")
        for name in ("eval_every", "eval_batches", "save_every", "sample_every"):
            if getattr(self, name) < 0:
                raise ConfigError(f"{name} must be >= 0, got {getattr(self, name)}")
        if self.seq_len is not None and self.seq_len <= 0:
            raise ConfigError(f"seq_len must be positive, got {self.seq_len}")
        check_memory_limit(self.cuda_memory_limit_gib)
        if not self.run_name or Path(self.run_name).name != self.run_name:
            raise ConfigError(f"run_name must be a plain directory name, got {self.run_name!r}")

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: dict[str, Any] | None = None) -> TrainConfig:
        """Load a run config; ``overrides`` keys may be dotted (``optim.lr``) to reach nested fields."""
        return cls.from_dict(load_yaml(path), overrides, source=str(path))

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], overrides: dict[str, Any] | None = None, source: str = "config"
    ) -> TrainConfig:
        return build_with_overrides(cls, data, overrides, source)

    @property
    def run_dir(self) -> Path:
        return Path(self.out_dir) / self.run_name

    @property
    def tokens_per_step(self) -> int:
        """Needs ``seq_len`` resolved (``resolve_model_config`` does that)."""
        if self.seq_len is None:
            raise ValueError("seq_len is not resolved yet")
        return self.micro_batch * self.grad_accum * self.seq_len


def check_memory_limit(limit: float | None) -> None:
    if limit is not None and limit <= 0:
        raise ConfigError(f"cuda_memory_limit_gib must be positive, got {limit}")


def build_with_overrides(
    cls: type[T], data: dict[str, Any], overrides: dict[str, Any] | None = None, source: str = "config"
) -> T:
    """Build config dataclass ``cls`` from ``data`` after applying ``key=value`` overrides.

    Override keys may be dotted (``optim.lr``) to reach nested settings; sibling
    keys of a nested mapping are kept.
    """
    data = copy.deepcopy(data)
    for key, value in (overrides or {}).items():
        *parents, leaf = key.split(".")
        node = data
        for p in parents:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                raise ConfigError(f"override {key}: {p} is not a mapping")
        node[leaf] = value
    try:
        return from_dict(cls, data)
    except ConfigError as e:
        raise ConfigError(f"{source}: {e}") from None


def resolve_model_config(cfg: TrainConfig) -> tuple[TrainConfig, ModelConfig]:
    """Load the model config named by ``cfg.model`` and make ``seq_len`` and context length agree."""
    path = Path(cfg.model)
    model_cfg = (
        ModelConfig.from_yaml(path) if path.suffix in (".yaml", ".yml") else ModelConfig.preset(cfg.model)
    )
    seq_len = cfg.seq_len or model_cfg.context_length
    model_cfg = dataclasses.replace(model_cfg, context_length=seq_len)
    return dataclasses.replace(cfg, seq_len=seq_len), model_cfg
