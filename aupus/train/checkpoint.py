"""Training checkpoints: everything needed to continue a run exactly where it stopped.

A checkpoint (``step_NNNNNN.pt`` in the run directory) holds:

* model weights and the AdamW state (first/second moments per parameter);
* the GradScaler state (fp16 runs only);
* the step number. The learning rate is a function of the step, so no
  scheduler state is needed;
* the dataloader position ``(epoch, position)``, so no batch is repeated or skipped;
* RNG states (Python, torch CPU, torch CUDA), so text samples continue the same sequence;
* the resolved train and model configs, and the best validation loss so far.

Resuming from step N and training to step M gives the same weights as one
uninterrupted run to M (``tests/test_train.py`` checks this bit for bit on CPU).

**Safety.** Files are loaded with ``torch.load(weights_only=True)``, which only
rebuilds tensors and plain containers and never runs pickled code. Saves go to
a temporary file first and are then renamed, so a crash during a save never
leaves a truncated checkpoint behind.

Training checkpoints are fp32 and include optimizer state (~3x the weights).
The release export (Milestone 11) writes weights only, in safetensors format.
"""

from __future__ import annotations

import os
import random
import re
from pathlib import Path
from typing import Any

import torch

_STEP_FILE = re.compile(r"^step_(\d{6,})\.pt$")
BEST = "best.pt"


def step_path(run_dir: str | Path, step: int) -> Path:
    return Path(run_dir) / f"step_{step:06d}.pt"


def list_checkpoints(run_dir: str | Path) -> list[tuple[int, Path]]:
    """``(step, path)`` of every step checkpoint in ``run_dir``, oldest first."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        return []
    found = [(int(m.group(1)), p) for p in run_dir.iterdir() if (m := _STEP_FILE.match(p.name))]
    return sorted(found)


def latest_checkpoint(run_dir: str | Path) -> Path | None:
    found = list_checkpoints(run_dir)
    return found[-1][1] if found else None


def rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict[str, Any]) -> None:
    # weights_only loading turns the Python RNG tuple into lists; random.setstate wants tuples.
    version, internal, gauss = state["python"]
    random.setstate((version, tuple(internal), gauss))
    # RNG states must be CPU byte tensors, but loading with map_location="cuda" moves them to the GPU.
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available() and len(state["cuda"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def save(state: dict[str, Any], path: str | Path) -> None:
    """Atomic ``torch.save``: write ``<path>.tmp``, then rename over ``path``."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=True)


def prune(run_dir: str | Path, keep_last: int) -> list[Path]:
    """Delete all but the newest ``keep_last`` step checkpoints (``best.pt`` is never touched)."""
    removed = []
    for _, path in list_checkpoints(run_dir)[:-keep_last]:
        path.unlink()
        removed.append(path)
    return removed
