"""Data pipeline: normalise, filter, dedup, tokenize into uint16 shards; deterministic resumable loader.

The loader is imported lazily: it needs torch, and the pipeline's worker
processes (which import this package) shouldn't pay torch's import time and memory.
"""

from typing import Any

from aupus.data.config import DataConfig, DedupConfig, FilterConfig
from aupus.data.dedup import Deduplicator, MinHasher, estimated_jaccard, exact_hash
from aupus.data.filters import check
from aupus.data.normalize import normalize
from aupus.data.pipeline import iter_fineweb_edu, iter_jsonl, prepare
from aupus.data.shards import ShardWriter, check_tokenizer, is_val, open_split

__all__ = [
    "DataConfig",
    "DedupConfig",
    "Deduplicator",
    "FilterConfig",
    "MinHasher",
    "PretrainLoader",
    "ShardWriter",
    "check",
    "check_tokenizer",
    "estimated_jaccard",
    "eval_batches",
    "exact_hash",
    "is_val",
    "iter_fineweb_edu",
    "iter_jsonl",
    "normalize",
    "open_split",
    "prepare",
]


def __getattr__(name: str) -> Any:
    if name in ("PretrainLoader", "eval_batches"):
        from aupus.data import loader

        return getattr(loader, name)
    raise AttributeError(f"module 'aupus.data' has no attribute {name!r}")
