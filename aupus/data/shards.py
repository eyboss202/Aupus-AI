"""Token shards on disk: flat ``uint16`` files plus a JSON index.

Layout of a prepared dataset directory::

    meta.json            format, tokenizer fingerprint, list of shards per split
    train_00000.bin      raw little-endian uint16 token ids, documents joined by <|eos|>
    train_00001.bin
    val_00000.bin

Why this format: a 32,000-id vocab fits in 16 bits, so 2.5B tokens are 5 GB.
Raw arrays can be memory-mapped (``np.memmap``): the dataloader reads any
window directly from disk, nothing is loaded into RAM up front, and the OS page
cache does the caching. ``meta.json`` records the tokenizer's SHA-256, so
training refuses to run on shards made with a different tokenizer.

Every document ends with ``<|eos|>``. During pretraining the model therefore
learns where documents stop, and generation can stop on that token.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

FORMAT = "aupus-shards"
VERSION = 1
DTYPE = np.dtype("<u2")  # little-endian uint16 on every platform


def is_val(digest: bytes, fraction: float) -> bool:
    """Deterministic split: a document is in val iff its hash falls in the lowest ``fraction``.

    Hash-based (not position-based), so a document's split never depends on
    download order, worker count or which other documents were filtered.
    """
    return int.from_bytes(digest[:8], "little") < fraction * 2**64


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class ShardWriter:
    """Appends documents' token arrays to ``<prefix>_NNNNN.bin`` files of ~``shard_tokens`` each."""

    def __init__(self, out_dir: str | Path, prefix: str, shard_tokens: int) -> None:
        self.out_dir = Path(out_dir)
        self.prefix = prefix
        self.shard_tokens = shard_tokens
        self.shards: list[dict[str, Any]] = []
        self._buffer: list[np.ndarray] = []
        self._buffered = 0
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def add(self, tokens: np.ndarray) -> None:
        """Add one document (tokens must already end with <|eos|>)."""
        self._buffer.append(np.asarray(tokens, dtype=DTYPE))
        self._buffered += len(tokens)
        if self._buffered >= self.shard_tokens:
            self._flush()

    def _flush(self) -> None:
        if not self._buffered:
            return
        name = f"{self.prefix}_{len(self.shards):05d}.bin"
        data = np.concatenate(self._buffer)
        data.tofile(self.out_dir / name)
        self.shards.append({"file": name, "tokens": int(len(data))})
        self._buffer, self._buffered = [], 0

    def close(self) -> list[dict[str, Any]]:
        self._flush()
        return self.shards


def write_meta(
    out_dir: str | Path,
    splits: dict[str, list[dict[str, Any]]],
    tokenizer_path: str | Path,
    vocab_size: int,
    eos_id: int,
    extra: dict[str, Any] | None = None,
) -> Path:
    meta = {
        "format": FORMAT,
        "version": VERSION,
        "dtype": "uint16",
        "tokenizer": Path(tokenizer_path).as_posix(),
        "tokenizer_sha256": file_sha256(tokenizer_path),
        "vocab_size": vocab_size,
        "eos_id": eos_id,
        "splits": splits,
        **(extra or {}),
    }
    path = Path(out_dir) / "meta.json"
    path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return path


@dataclass
class Shard:
    path: Path
    tokens: np.ndarray  # memory-mapped uint16 array


def open_split(data_dir: str | Path, split: str) -> tuple[dict[str, Any], list[Shard]]:
    """Memory-map every shard of ``split``; checks the format and that file sizes match the index."""
    data_dir = Path(data_dir)
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    if meta.get("format") != FORMAT or meta.get("version") != VERSION:
        raise ValueError(f"{data_dir}: not an {FORMAT} v{VERSION} dataset")
    if split not in meta["splits"]:
        raise ValueError(f"{data_dir}: no split {split!r} (have {sorted(meta['splits'])})")
    shards = []
    for entry in meta["splits"][split]:
        path = data_dir / entry["file"]
        size = path.stat().st_size
        if size != entry["tokens"] * DTYPE.itemsize:
            raise ValueError(f"{path}: {size} bytes, expected {entry['tokens']} tokens")
        shards.append(Shard(path, np.memmap(path, dtype=DTYPE, mode="r")))
    return meta, shards


def check_tokenizer(meta: dict[str, Any], tokenizer_path: str | Path) -> None:
    """Raise if the shards were made with a different tokenizer file."""
    actual = file_sha256(tokenizer_path)
    if actual != meta["tokenizer_sha256"]:
        raise ValueError(
            f"shards were tokenized with {meta['tokenizer']} (sha256 {meta['tokenizer_sha256'][:12]}), "
            f"but {tokenizer_path} has sha256 {actual[:12]}"
        )
