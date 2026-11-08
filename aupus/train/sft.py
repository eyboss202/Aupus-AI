"""Supervised fine-tuning (SFT): turn a pretrained Base model into a chat model (Milestone 10).

Pretraining teaches a model to continue web text. SFT keeps training the same
weights on conversations in the chat template, with the loss only on the
assistant's replies. The model learns the turn format, to answer rather than
ramble, and to stop with ``<|end|>``. It learns very little new knowledge: there
are ~3M reply tokens here against billions in pretraining.

It is the same loop as pretraining (``Trainer``: AdamW, warmup + cosine,
gradient accumulation, checkpoints, exact resume, Ctrl-C), with these changes:

* the weights start from a Base checkpoint (``init_from``), and the model shape
  comes from that checkpoint;
* the data are padded conversations (``aupus/data/sft.py``), not packed windows;
* the learning rate is ~20x lower than pretraining's peak (3e-5 vs 6e-4).
  The model is already good at language, and big updates on a small dataset
  would wreck what it knows ("catastrophic forgetting");
* training runs for a number of **epochs** over the conversations (2-3 is the
  usual range), with **early stopping**: the run ends once validation loss hasn't
  improved for ``patience`` evaluations in a row. ``best.pt`` keeps the best one.
  A small dataset is easy to overfit: train loss keeps falling while the model
  memorises replies, and val loss starts rising.

One detail: with gradient accumulation, each micro-batch's loss is a mean over
*its* reply tokens, and micro-batches are averaged with equal weight. A
micro-batch of short replies therefore counts as much as one of long replies.
That is the usual simplification and it doesn't matter in practice.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from aupus.config import ConfigError, load_yaml
from aupus.data.sft import SFTDataset, SFTLoader, pad_id, read_jsonl, sft_eval_batches, tokenize
from aupus.data.shards import file_sha256
from aupus.model import ModelConfig
from aupus.tokenizer import END, Tokenizer
from aupus.tokenizer.chat import prompt_ids
from aupus.train.config import DEVICES, PRECISIONS, OptimConfig, build_with_overrides
from aupus.train.pretrain import Trainer


@dataclass(frozen=True)
class SFTOptimConfig(OptimConfig):
    """``OptimConfig`` with fine-tuning defaults, so a partial ``optim:`` section keeps them."""

    lr: float = 3e-5
    min_lr: float = 3e-6
    warmup_steps: int = 20


@dataclass(frozen=True)
class SFTConfig:
    """One fine-tuning run. Keys shared with ``TrainConfig`` mean the same thing there.

    Attributes:
        init_from: the Base checkpoint to start from (usually ``runs/<base run>/best.pt``).
        data_dir: directory with ``train.jsonl`` and ``val.jsonl`` (``scripts/prepare_sft_data.py``).
        seq_len: max tokens per conversation; longer ones are cut. ``None`` = the Base
            model's context length (it can't be longer: positions beyond it were never trained).
        epochs: passes over the training conversations; sets ``max_steps`` when that is ``None``.
        max_steps: optimizer steps. ``None`` = computed from ``epochs`` (the resolved value
            is saved in the run's ``config.yaml``).
        patience: stop after this many evaluations in a row without a new best val loss (0 = never).
    """

    run_name: str
    init_from: str
    data_dir: str = "data/sft"
    tokenizer: str = "data/tokenizer/aupus-bpe-32k.json"
    out_dir: str = "runs"
    seq_len: int | None = None
    micro_batch: int = 8
    grad_accum: int = 4
    epochs: float = 3.0
    max_steps: int | None = None
    patience: int = 3
    optim: SFTOptimConfig = field(default_factory=SFTOptimConfig)
    device: str = "auto"
    precision: str = "auto"
    compile: bool = True
    grad_checkpointing: bool = False
    eval_every: int = 100
    save_every: int = 200
    keep_last: int = 2
    log_every: int = 10
    sample_every: int = 0
    sample_prompts: list[str] = field(default_factory=list)
    sample_tokens: int = 128
    seed: int = 1337

    def __post_init__(self) -> None:
        if self.device not in DEVICES:
            raise ConfigError(f"device must be one of {DEVICES}, got {self.device!r}")
        if self.precision not in PRECISIONS:
            raise ConfigError(f"precision must be one of {PRECISIONS}, got {self.precision!r}")
        for name in ("micro_batch", "grad_accum", "log_every", "keep_last", "sample_tokens"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"{name} must be positive, got {getattr(self, name)}")
        for name in ("eval_every", "save_every", "sample_every", "patience"):
            if getattr(self, name) < 0:
                raise ConfigError(f"{name} must be >= 0, got {getattr(self, name)}")
        if self.epochs <= 0:
            raise ConfigError(f"epochs must be positive, got {self.epochs}")
        for name in ("seq_len", "max_steps"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ConfigError(f"{name} must be positive, got {value}")
        if not self.run_name or Path(self.run_name).name != self.run_name:
            raise ConfigError(f"run_name must be a plain directory name, got {self.run_name!r}")
        if not self.init_from:
            raise ConfigError("init_from must name the Base checkpoint to fine-tune")
        if self.patience and not self.eval_every:
            raise ConfigError("patience needs eval_every > 0 (early stopping is decided at evaluations)")

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: dict[str, Any] | None = None) -> SFTConfig:
        return cls.from_dict(load_yaml(path), overrides, source=str(path))

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], overrides: dict[str, Any] | None = None, source: str = "config"
    ) -> SFTConfig:
        return build_with_overrides(cls, data, overrides, source)

    @property
    def run_dir(self) -> Path:
        return Path(self.out_dir) / self.run_name

    @property
    def model(self) -> str:
        """What the run log prints as the model: where its weights came from."""
        return f"{self.init_from} (fine-tuned)"


class SFTTrainer(Trainer):
    """``Trainer`` with SFT data, Base weights, chat-formatted samples and early stopping."""

    stage = "sft"

    def __init__(self, cfg: SFTConfig, resume: bool = False, log: Callable[[str], None] = print) -> None:
        super().__init__(cfg, resume=resume, log=log)  # type: ignore[arg-type]  # duck-typed config

    # ------------------------------------------------------------ hooks

    def _resolve_configs(self, cfg: SFTConfig) -> tuple[SFTConfig, ModelConfig]:  # type: ignore[override]
        base = torch.load(cfg.init_from, map_location="cpu", weights_only=True, mmap=True)
        model_cfg = ModelConfig(**base["model_config"])
        expected = base.get("tokenizer_sha256")
        if expected is not None and file_sha256(cfg.tokenizer) != expected:
            raise ValueError(f"{cfg.tokenizer} is not the tokenizer {cfg.init_from} was trained with")
        seq_len = cfg.seq_len or model_cfg.context_length
        if seq_len > model_cfg.context_length:
            raise ConfigError(
                f"seq_len {seq_len} is longer than the Base model's context ({model_cfg.context_length})"
            )
        self.tokenizer = Tokenizer.load(cfg.tokenizer)
        data = Path(cfg.data_dir)
        self.train_data = tokenize(self.tokenizer, read_jsonl(data / "train.jsonl"), seq_len)
        self.val_data = tokenize(self.tokenizer, read_jsonl(data / "val.jsonl"), seq_len)
        if len(self.val_data) == 0 and cfg.eval_every:
            raise ValueError(f"{data / 'val.jsonl'} has no usable conversations")
        max_steps = cfg.max_steps
        if max_steps is None:
            batches = math.ceil(len(self.train_data) / cfg.micro_batch)
            max_steps = max(1, math.ceil(cfg.epochs * batches / cfg.grad_accum))
        resolved = dataclasses.replace(cfg, seq_len=seq_len, max_steps=max_steps)
        return resolved, dataclasses.replace(model_cfg, context_length=seq_len)

    def _setup_data(self) -> None:
        pass  # tokenizer and conversations were loaded in _resolve_configs

    def _init_weights(self) -> None:
        # mmap: read the model tensors only, not the Base run's optimizer state.
        state = torch.load(self.cfg.init_from, map_location="cpu", weights_only=True, mmap=True)
        self.model.load_state_dict(state["model"])
        stage, val = state.get("stage", "pretrain"), state.get("best_val_loss", math.nan)
        self.log(f"initialised from {self.cfg.init_from} ({stage} step {state['step']}, val loss {val:.4f})")

    def _make_loader(self) -> SFTLoader:
        return SFTLoader(self.train_data, self.cfg.micro_batch, pad_id(self.tokenizer), self.cfg.seed)

    def _sample_prompt(self, prompt: str) -> tuple[list[int], list[int]]:
        ids = prompt_ids(self.tokenizer, [{"role": "user", "content": prompt}])
        return ids, [self.tokenizer.special_id(END), self.eos_id]

    def _format_sample(self, prompt: str, generated: str) -> str:
        return f"User: {prompt}\nAupus: {generated}"

    def _describe_data(self) -> str:
        cfg, tr = self.cfg, self.train_data
        epochs = cfg.max_steps * cfg.grad_accum / self.loader.batches_per_epoch
        return (
            f"{len(tr):,} train / {len(self.val_data):,} val conversations "
            f"({tr.target_tokens / 1e6:.2f}M reply tokens; {tr.truncated} cut at {cfg.seq_len}, "
            f"{tr.dropped} dropped) | batch {cfg.micro_batch} x {cfg.grad_accum} accum | "
            f"{cfg.max_steps:,} steps = {epochs:.2f} epochs | early-stop patience {cfg.patience}"
        )

    def _data_metrics(self) -> dict[str, float]:
        return {"data/epochs": self.step * self.cfg.grad_accum / self.loader.batches_per_epoch}

    # ------------------------------------------------------------ eval + early stopping

    @torch.no_grad()
    def evaluate(self) -> float:
        """Mean loss per reply token over the whole val set."""
        pad = pad_id(self.tokenizer)
        return sft_loss(self.model, self.val_data, self.cfg.micro_batch, pad, self._autocast)

    def _do_eval(self) -> None:
        super()._do_eval()
        if self.cfg.patience and self.evals_since_best >= self.cfg.patience:
            self._halt_reason = (
                f"early stop: val loss hasn't improved for {self.evals_since_best} evaluations "
                f"(best {self.best_val_loss:.4f}, kept in best.pt)"
            )


@torch.no_grad()
def sft_loss(
    model: torch.nn.Module,
    data: SFTDataset,
    batch_size: int,
    pad: int,
    autocast: Callable[[], Any] | None = None,
) -> float:
    """Token-weighted mean cross-entropy over the reply tokens of ``data``."""
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    try:
        for x, y in sft_eval_batches(data, batch_size, pad):
            x, y = x.to(device), y.to(device)
            if autocast is not None:
                with autocast():
                    logits = model(x).logits
            else:
                logits = model(x).logits
            total += float(F.cross_entropy(logits.float().flatten(0, 1), y.flatten(), reduction="sum"))
            count += int((y >= 0).sum())
    finally:
        model.train(was_training)
    return total / max(count, 1)
