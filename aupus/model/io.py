"""Load a trained model for inference, and the weights-only release format.

A model exists on disk in one of two forms:

* a **training checkpoint** (``runs/<name>/best.pt`` or ``step_NNNNNN.pt``): weights plus
  optimizer state, loader position, RNG and configs (see ``aupus/train/checkpoint.py``).
  It's what training writes and what ``--resume`` needs, but it's ~3x bigger than the
  weights and only PyTorch can read it;
* a **release directory** (written by ``export_release``)::

      model.safetensors   weights only (fp32, fp16 or bf16)
      model.yaml          the ModelConfig
      tokenizer.json      the exact tokenizer the model was trained with
      release.json        where it came from: source checkpoint, step, val loss, SHA-256 of each file

Why safetensors for releases: a ``.safetensors`` file is a JSON header plus raw
tensor bytes. Loading it can't run code (a pickle-based ``.pt`` file can, which
is why we only ever load those with ``weights_only=True``), it can be memory-mapped,
and other tools can read it. ``safetensors`` is used for serialization only, not
for anything model-related.

``load_model`` accepts either form, so evaluation and chat work the same way on a
checkpoint from a run in progress and on a finished release.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from aupus.config import load_config, save_config
from aupus.data.shards import file_sha256
from aupus.model.config import ModelConfig, param_count
from aupus.model.transformer import Transformer
from aupus.tokenizer import Tokenizer

RELEASE_FORMAT = "aupus-release"
RELEASE_VERSION = 1
WEIGHTS = "model.safetensors"
MODEL_CONFIG = "model.yaml"
TOKENIZER = "tokenizer.json"
INFO = "release.json"
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


@dataclass
class LoadedModel:
    """A model ready for inference (eval mode, no gradients needed) and its tokenizer."""

    model: Transformer
    tokenizer: Tokenizer
    source: Path
    info: dict[str, Any] = field(default_factory=dict)  # step, val loss, ... when known


def _check_vocab(tokenizer: Tokenizer, cfg: ModelConfig, source: Path) -> None:
    if tokenizer.vocab_size > cfg.vocab_size:
        raise ValueError(
            f"{source}: tokenizer has {tokenizer.vocab_size} ids but the model only {cfg.vocab_size}"
        )


def _load_checkpoint(
    path: Path, tokenizer_path: str | Path | None
) -> tuple[ModelConfig, dict[str, torch.Tensor], Path, dict[str, Any]]:
    # mmap: only the model tensors are read from disk, not the (twice as big) optimizer state.
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if "model" not in state or "model_config" not in state:
        raise ValueError(f"{path} is not an Aupus training checkpoint")
    cfg = ModelConfig(**state["model_config"])
    train_cfg = state.get("train_config", {})
    tok = Path(tokenizer_path or train_cfg.get("tokenizer", "data/tokenizer/aupus-bpe-32k.json"))
    expected = state.get("tokenizer_sha256")  # recorded since Milestone 10
    if expected is not None and file_sha256(tok) != expected:
        raise ValueError(f"{tok} is not the tokenizer {path} was trained with (SHA-256 differs)")
    info = {
        "kind": "checkpoint",
        "step": state.get("step"),
        "best_val_loss": state.get("best_val_loss"),
        "run_name": train_cfg.get("run_name"),
        "stage": state.get("stage", "pretrain"),
    }
    return cfg, state["model"], tok, info


def _load_release(
    path: Path, tokenizer_path: str | Path | None
) -> tuple[ModelConfig, dict[str, torch.Tensor], Path, dict[str, Any]]:
    from safetensors.torch import load_file

    info = json.loads((path / INFO).read_text(encoding="utf-8"))
    if info.get("format") != RELEASE_FORMAT or info.get("version") != RELEASE_VERSION:
        raise ValueError(f"{path}: not an {RELEASE_FORMAT} v{RELEASE_VERSION} directory")
    cfg = load_config(ModelConfig, path / MODEL_CONFIG)
    tok = Path(tokenizer_path) if tokenizer_path else path / TOKENIZER
    return cfg, load_file(path / WEIGHTS), tok, {"kind": "release", **info}


def load_model(
    path: str | Path,
    device: str | torch.device = "cpu",
    dtype: str = "fp32",
    tokenizer_path: str | Path | None = None,
) -> LoadedModel:
    """Load a training checkpoint (``*.pt``) or a release directory for inference.

    Args:
        path: ``runs/<name>/best.pt`` (or any step checkpoint), or a release directory.
        device: where to put the model.
        dtype: compute dtype. Releases may be stored in fp16/bf16, but on CPU (and on
            GPUs without fast half precision, like the GTX 1650) fp32 is faster, so
            weights are converted to ``dtype`` after loading.
        tokenizer_path: override the tokenizer (default: the one recorded with the model).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"model not found: {path} (relative paths start from the current folder, {Path.cwd()}; "
            "run the scripts from the repository folder, e.g. `python scripts/chat_web.py --model ...`)"
        )
    loader = _load_release if path.is_dir() else _load_checkpoint
    cfg, weights, tok_path, info = loader(path, tokenizer_path)
    tokenizer = Tokenizer.load(tok_path)
    _check_vocab(tokenizer, cfg, path)
    model = Transformer(cfg)
    model.load_state_dict({k: v.float() for k, v in weights.items()})  # strict: every key must match
    model.to(device=device, dtype=DTYPES[dtype]).eval()
    return LoadedModel(model, tokenizer, path, info)


def export_release(
    checkpoint_path: str | Path,
    out_dir: str | Path,
    dtype: str = "fp32",
    tokenizer_path: str | Path | None = None,
    overwrite: bool = False,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a weights-only release directory from a training checkpoint.

    ``fp16``/``bf16`` halve the download (125M params: ~500 MB in fp32, ~250 MB in
    half precision). bf16 keeps fp32's range, so it can't overflow, but has fewer
    mantissa bits; fp16 is the other way round. The tied embedding is stored once.
    """
    from safetensors.torch import save_file

    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {sorted(DTYPES)}, got {dtype!r}")
    checkpoint_path, out_dir = Path(checkpoint_path), Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{out_dir} is not empty (pass overwrite=True / --force to replace it)")
    cfg, weights, tok_path, info = _load_checkpoint(checkpoint_path, tokenizer_path)
    Tokenizer.load(tok_path)  # fail early on a broken tokenizer file
    out_dir.mkdir(parents=True, exist_ok=True)

    # .contiguous(): safetensors stores raw bytes, so every tensor must be one dense block.
    tensors = {k: v.to(DTYPES[dtype]).contiguous() for k, v in weights.items()}
    save_file(tensors, out_dir / WEIGHTS, metadata={"format": RELEASE_FORMAT, "dtype": dtype})
    save_config(cfg, out_dir / MODEL_CONFIG)
    shutil.copyfile(tok_path, out_dir / TOKENIZER)
    release = {
        "format": RELEASE_FORMAT,
        "version": RELEASE_VERSION,
        "dtype": dtype,
        "params": param_count(cfg)["total"],
        "source_checkpoint": checkpoint_path.as_posix(),
        "step": info["step"],
        "best_val_loss": info["best_val_loss"],
        "run_name": info["run_name"],
        "stage": info["stage"],
        "sha256": {name: file_sha256(out_dir / name) for name in (WEIGHTS, MODEL_CONFIG, TOKENIZER)},
        **(extra or {}),
    }
    (out_dir / INFO).write_text(json.dumps(release, indent=2) + "\n", encoding="utf-8")
    return out_dir
