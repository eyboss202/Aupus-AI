"""Deterministic, resumable batches of training sequences from token shards.

Each shard is cut into non-overlapping windows of ``seq_len + 1`` tokens that
share one token at the edges (window ``i`` covers ``[i*seq_len, (i+1)*seq_len]``).
Input is ``window[:-1]`` and target is ``window[1:]``, so every token in the
shard is predicted exactly once per epoch. Windows don't cross shard
boundaries, which loses at most ``seq_len`` tokens per 100M-token shard.

Windows ignore document boundaries ("packing"): a window can hold the end of
one document, ``<|eos|>``, and the start of the next. Nothing is wasted on
padding, and the model learns that text after ``<|eos|>`` is unrelated. This is
standard for GPT-style pretraining.

**Determinism and resume.** The window order for epoch ``e`` is a permutation
drawn from ``np.random.default_rng([seed, e])``. The loader's whole state is
``(epoch, position)``, two integers saved in every checkpoint. After a crash
the loader regenerates the same permutation and continues at exactly the next
batch, with no repeated or skipped data.

``position`` counts windows, not batches, so a run may resume with a different
micro-batch size (for example a 24 GB rented GPU at 16 sequences per batch,
then a 4 GB card at 1): the stream of windows stays the same, only how it is
cut into batches changes. The seed, ``seq_len`` and the dataset must match.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch

from aupus.data.shards import Shard, open_split


class _Windows:
    """Index -> (shard, offset) mapping over all windows of a split."""

    def __init__(self, shards: list[Shard], seq_len: int) -> None:
        self.shards = shards
        self.seq_len = seq_len
        counts = [max(0, (len(s.tokens) - 1) // seq_len) for s in shards]
        self.starts = np.cumsum([0, *counts])
        self.total = int(self.starts[-1])
        if self.total == 0:
            raise ValueError(f"no complete window of {seq_len + 1} tokens in this split")

    def get(self, idx: np.ndarray) -> torch.Tensor:
        """Stack windows ``idx`` into an int64 tensor of shape (len(idx), seq_len + 1)."""
        out = np.empty((len(idx), self.seq_len + 1), dtype=np.int64)
        shard_ids = np.searchsorted(self.starts, idx, side="right") - 1
        for row, (i, s) in enumerate(zip(idx, shard_ids, strict=True)):
            offset = int(i - self.starts[s]) * self.seq_len
            out[row] = self.shards[s].tokens[offset : offset + self.seq_len + 1]
        return torch.from_numpy(out)


class PretrainLoader:
    """Infinite iterator of ``(inputs, targets)`` int64 batches, each (batch_size, seq_len).

    Args:
        data_dir: dataset directory with ``meta.json``.
        split: "train" or "val".
        batch_size: sequences per batch (the micro-batch).
        seq_len: tokens per sequence (the model's context length or less).
        seed: base seed of the per-epoch permutation.
    """

    def __init__(self, data_dir: str | Path, split: str, batch_size: int, seq_len: int, seed: int = 0) -> None:
        self.meta, shards = open_split(data_dir, split)
        self.windows = _Windows(shards, seq_len)
        self.batch_size, self.seq_len, self.seed = batch_size, seq_len, seed
        self.epoch, self.position = 0, 0
        self._order = self._permutation(0)

    @property
    def num_windows(self) -> int:
        return self.windows.total

    def _permutation(self, epoch: int) -> np.ndarray:
        return np.random.default_rng([self.seed, epoch]).permutation(self.windows.total)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        idx = []
        while len(idx) < self.batch_size:
            if self.position == len(self._order):  # epoch boundary, possibly mid-batch
                self.epoch += 1
                self.position = 0
                self._order = self._permutation(self.epoch)
            take = min(self.batch_size - len(idx), len(self._order) - self.position)
            idx.extend(self._order[self.position : self.position + take])
            self.position += take
        batch = self.windows.get(np.asarray(idx))
        return batch[:, :-1], batch[:, 1:]

    def state_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "position": self.position,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "num_windows": self.windows.total,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        # batch_size is saved but may differ (see the module docstring): the trainer checks
        # that each optimizer step still sees the same number of sequences.
        for key in ("seed", "seq_len", "num_windows"):
            mine = self.state_dict()[key]
            if state[key] != mine:
                raise ValueError(f"loader state mismatch for {key}: checkpoint {state[key]}, current {mine}")
        self.epoch, self.position = state["epoch"], state["position"]
        self._order = self._permutation(self.epoch)


def eval_batches(
    data_dir: str | Path, split: str, batch_size: int, seq_len: int, max_batches: int | None = None
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """One ordered pass over a split (validation loss must use the same tokens every time)."""
    _, shards = open_split(data_dir, split)
    windows = _Windows(shards, seq_len)
    n = windows.total if max_batches is None else min(windows.total, max_batches * batch_size)
    for start in range(0, n, batch_size):
        batch = windows.get(np.arange(start, min(start + batch_size, n)))
        yield batch[:, :-1], batch[:, 1:]
