"""The pretraining data pipeline: raw documents -> token shards.

    source -> normalize -> quality filters -> exact dedup -> near dedup -> train/val split
           -> tokenize + <|eos|> -> uint16 shards + meta.json + stats.json

Work split:
  * **worker processes** do everything per document that doesn't depend on
    other documents: normalise, filter, hash, MinHash signature, tokenize.
    Results come back in input order (``Pool.imap``);
  * **the main process** does what needs global state, in order: the dedup
    sets, the split, and appending to shards.

So the output is identical for any number of workers. Documents are
tokenized before dedup is known, which wastes the work on duplicates (a few
percent) but keeps the pipeline a single streaming pass.
"""

from __future__ import annotations

import json
import multiprocessing
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from aupus.config import to_dict
from aupus.data.config import DataConfig
from aupus.data.dedup import Deduplicator, MinHasher, exact_hash
from aupus.data.filters import check
from aupus.data.normalize import normalize
from aupus.data.shards import DTYPE, ShardWriter, is_val, write_meta
from aupus.tokenizer import EOS, Tokenizer

# ------------------------------------------------------------------ sources


def iter_jsonl(path: str | Path, field: str = "text") -> Iterator[str]:
    """Yield ``record[field]`` for each line of a JSON-lines file."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)[field]


FINEWEB_EDU = "HuggingFaceFW/fineweb-edu"
FINEWEB_EDU_SUBSET = "sample-10BT"
FINEWEB_EDU_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"


def iter_parquet(paths: Iterable[str | Path], field: str = "text", batch_rows: int = 1024) -> Iterator[str]:
    """Yield the ``field`` column of each parquet file in turn, row by row.

    Files are read in the order given and rows in file order, which is the same
    order ``datasets`` streams them in. That way a local copy of FineWeb-Edu gives
    the same documents in the same order as :func:`iter_fineweb_edu`. Only the
    one column is read, ``batch_rows`` rows at a time, so memory stays small
    even though each file is ~2 GB. Needs ``pyarrow`` (installed with ``.[data]``).
    """
    import pyarrow.parquet as pq

    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_rows, columns=[field]):
            yield from batch.column(0).to_pylist()


def iter_fineweb_edu(revision: str = FINEWEB_EDU_REVISION) -> Iterator[str]:
    """Stream FineWeb-Edu documents (needs the optional ``datasets`` dependency).

    Fine for small samples. For the full Milestone 7 set, download the parquet
    files first (``scripts/download_fineweb_edu.py``) and use :func:`iter_parquet`:
    a dropped connection then costs one resumed file, not the whole run.
    """
    from datasets import load_dataset

    for record in load_dataset(FINEWEB_EDU, name=FINEWEB_EDU_SUBSET, split="train", streaming=True, revision=revision):
        yield record["text"]


# ------------------------------------------------------------------ worker side


@dataclass
class DocResult:
    reason: str | None  # filter rule that rejected the doc, or None
    digest: bytes = b""
    band_keys: list[int] | None = None
    tokens: np.ndarray | None = None
    raw_bytes: int = 0


_worker: dict[str, Any] = {}


def _init_worker(cfg_dict: dict[str, Any], tokenizer_path: str) -> None:
    from aupus.config import from_dict

    cfg = from_dict(DataConfig, cfg_dict)
    tok = Tokenizer.load(tokenizer_path)
    _worker.update(cfg=cfg, tok=tok, eos=tok.special_id(EOS), hasher=MinHasher(cfg.dedup))


def process_document(raw: str) -> DocResult:
    """Everything that can be done for one document in isolation."""
    cfg, tok = _worker["cfg"], _worker["tok"]
    raw_bytes = len(raw.encode("utf-8"))
    text = normalize(raw)
    reason = check(text, cfg.filters)
    if reason is not None:
        return DocResult(reason, raw_bytes=raw_bytes)
    hasher: MinHasher = _worker["hasher"]
    ids = tok.encode_ordinary(text)  # raw web text never produces special tokens
    ids.append(_worker["eos"])
    return DocResult(
        None,
        digest=exact_hash(text),
        band_keys=hasher.band_keys(hasher.signature(text)),
        tokens=np.asarray(ids, dtype=DTYPE),
        raw_bytes=raw_bytes,
    )


def _process_batch(batch: list[str]) -> list[DocResult]:
    return [process_document(t) for t in batch]


def _batched(texts: Iterable[str], size: int, limit: int | None) -> Iterator[list[str]]:
    batch: list[str] = []
    for i, text in enumerate(texts):
        if limit is not None and i >= limit:
            break
        batch.append(text)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


# ------------------------------------------------------------------ main side


def prepare(
    texts: Iterable[str],
    out_dir: str | Path,
    tokenizer_path: str | Path,
    cfg: DataConfig,
    num_workers: int = 1,
    max_docs: int | None = None,
    batch_size: int = 64,
    progress: Any = None,
) -> dict[str, Any]:
    """Run the whole pipeline; returns (and writes) the stats dict.

    Args:
        texts: raw documents, in a fixed order.
        out_dir: output directory (created; existing shards are overwritten).
        tokenizer_path: the tokenizer JSON to encode with (its hash goes in meta.json).
        cfg: filter / dedup / split / shard settings.
        num_workers: worker processes (1 = run in this process, handy for tests).
        max_docs: stop after this many input documents.
        progress: optional ``tqdm``-like object; ``update(n)`` is called per document.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in [*out_dir.glob("*.bin"), out_dir / "meta.json"]:
        old.unlink(missing_ok=True)  # never mix shards from two runs
    cfg_dict = to_dict(cfg)
    writers = {s: ShardWriter(out_dir, s, cfg.shard_tokens) for s in ("train", "val")}
    dedup = Deduplicator()
    counts: Counter[str] = Counter()
    tokens = Counter[str]()
    raw_bytes = kept_bytes = 0

    start = time.perf_counter()
    batches = _batched(texts, batch_size, max_docs)
    if num_workers <= 1:
        _init_worker(cfg_dict, str(tokenizer_path))
        results: Iterable[list[DocResult]] = map(_process_batch, batches)
        pool = None
    else:
        pool = multiprocessing.get_context("spawn").Pool(
            num_workers, initializer=_init_worker, initargs=(cfg_dict, str(tokenizer_path))
        )
        results = pool.imap(_process_batch, batches, chunksize=1)
    try:
        for batch in results:
            for r in batch:
                counts["input"] += 1
                raw_bytes += r.raw_bytes
                if r.reason is not None:
                    counts[f"filtered:{r.reason}"] += 1
                elif dedup.is_exact_duplicate(r.digest):
                    counts["dedup:exact"] += 1
                elif dedup.is_near_duplicate(r.band_keys or []):
                    counts["dedup:near"] += 1
                else:
                    split = "val" if is_val(r.digest, cfg.val_fraction) else "train"
                    assert r.tokens is not None
                    writers[split].add(r.tokens)
                    counts[f"kept:{split}"] += 1
                    tokens[split] += len(r.tokens)
                    kept_bytes += r.raw_bytes
            if progress is not None:
                progress.update(len(batch))
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    seconds = time.perf_counter() - start

    tok = Tokenizer.load(tokenizer_path)
    splits = {s: w.close() for s, w in writers.items()}
    write_meta(out_dir, splits, tokenizer_path, tok.vocab_size, tok.special_id(EOS), extra={"config": cfg_dict})
    n_in = counts["input"]
    kept = counts["kept:train"] + counts["kept:val"]
    stats = {
        "documents": dict(sorted(counts.items())),
        "kept_fraction": round(kept / n_in, 4) if n_in else 0.0,
        "tokens": dict(tokens),
        "raw_mb": round(raw_bytes / 1e6, 1),
        "kept_raw_mb": round(kept_bytes / 1e6, 1),
        "tokens_per_kept_byte": round(sum(tokens.values()) / kept_bytes, 4) if kept_bytes else 0.0,
        "seconds": round(seconds, 1),
        "docs_per_second": round(n_in / seconds, 1) if seconds else 0.0,
        "raw_mb_per_second": round(raw_bytes / 1e6 / seconds, 2) if seconds else 0.0,
        "num_workers": num_workers,
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return stats
