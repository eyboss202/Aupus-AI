"""Data pipeline: normalise, filter, dedup, tokenize into uint16 shards; deterministic resumable loader.

The loader is imported lazily: it needs torch, and the pipeline's worker
processes (which import this package) shouldn't pay torch's import time and memory.
"""

from typing import Any

from vulpes.data.config import DataConfig, DedupConfig, FilterConfig
from vulpes.data.dedup import Deduplicator, MinHasher, estimated_jaccard, exact_hash
from vulpes.data.filters import check
from vulpes.data.normalize import normalize
from vulpes.data.pipeline import iter_fineweb_edu, iter_jsonl, iter_parquet, prepare
from vulpes.data.shards import ShardWriter, check_tokenizer, is_val, open_split

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
    "iter_parquet",
    "normalize",
    "open_split",
    "prepare",
]


def __getattr__(name: str) -> Any:
    if name in ("PretrainLoader", "eval_batches"):
        from vulpes.data import loader

        return getattr(loader, name)
    raise AttributeError(f"module 'vulpes.data' has no attribute {name!r}")
