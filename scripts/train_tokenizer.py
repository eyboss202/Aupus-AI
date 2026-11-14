"""Train the Vulpes BPE tokenizer and report how good and how fast it is.

Steps (each timed):
  1. count distinct pre-tokenized chunks in the training JSONL (parallel);
  2. learn merges until the vocab has ``--vocab-size`` ids (incl. special tokens);
  3. save ``<out>.json`` (the tokenizer) and ``<out>.vocab.txt`` (human-readable);
  4. on held-out text: check lossless round-trip, measure compression
     (bytes/token, chars/token) and single-process encode speed.

The stats are also written to ``<out>.stats.json``, so the data-pipeline
decision in docs/PLAN.md (how long tokenizing 2.5B tokens will take) is based on
measured numbers.

Usage:
    python scripts/train_tokenizer.py                       # 32k vocab on the M3 sample
    python scripts/train_tokenizer.py --vocab-size 8192 --out data/tokenizer/vulpes-bpe-8k
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from collections.abc import Iterator
from pathlib import Path

from tqdm import tqdm

from vulpes.tokenizer import SPECIAL_TOKENS, Tokenizer, count_chunks, train_bpe


def read_jsonl(path: Path, max_docs: int | None = None) -> Iterator[str]:
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_docs is not None and i >= max_docs:
                return
            yield json.loads(line)["text"]


def main() -> None:
    fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=fmt)
    parser.add_argument("--input", type=Path, default=Path("data/tokenizer_sample/train.jsonl"))
    parser.add_argument("--heldout", type=Path, default=Path("data/tokenizer_sample/heldout.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/tokenizer/vulpes-bpe-32k"), help="no suffix")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--max-docs", type=int, default=None, help="only use the first N training docs")
    args = parser.parse_args()

    machine = f"{platform.processor() or platform.machine()}, {os.cpu_count()} threads"
    stats: dict[str, object] = {"machine": machine}
    num_merges = args.vocab_size - 256 - len(SPECIAL_TOKENS)

    t0 = time.perf_counter()
    train_bytes = args.input.stat().st_size if args.max_docs is None else None
    counts = count_chunks(read_jsonl(args.input, args.max_docs), num_workers=args.workers)
    stats["count_seconds"] = round(time.perf_counter() - t0, 1)
    stats["distinct_chunks"] = len(counts)
    stats["total_chunks"] = sum(counts.values())
    stats["train_file_bytes"] = train_bytes
    print(
        f"[1/4] counted {stats['total_chunks']:,} chunks ({len(counts):,} distinct) "
        f"in {stats['count_seconds']}s with {args.workers} workers"
    )

    t0 = time.perf_counter()
    with tqdm(total=num_merges, desc="[2/4] merges", unit="merge", mininterval=2.0) as bar:
        merges = train_bpe(
            counts, num_merges, min_frequency=args.min_frequency, progress=lambda n: bar.update(n - bar.n)
        )
    del counts
    tok = Tokenizer(merges)
    stats["merge_seconds"] = round(time.perf_counter() - t0, 1)
    stats["vocab_size"] = tok.vocab_size
    print(f"[2/4] learned {len(merges):,} merges in {stats['merge_seconds']}s -> {tok}")

    json_path = args.out.with_suffix(".json")
    tok.save(json_path)
    tok.dump_vocab(args.out.with_suffix(".vocab.txt"))
    print(f"[3/4] saved {json_path} and {args.out.with_suffix('.vocab.txt')}")

    loaded = Tokenizer.load(json_path)  # evaluate the saved file, not the in-memory object
    heldout = list(read_jsonl(args.heldout))
    n_bytes = sum(len(t.encode("utf-8")) for t in heldout)
    n_chars = sum(len(t) for t in heldout)
    t0 = time.perf_counter()
    encoded = [loaded.encode(t) for t in heldout]  # cold cache: realistic for the first shard
    encode_seconds = time.perf_counter() - t0
    n_tokens = sum(len(e) for e in encoded)
    roundtrip_ok = all(loaded.decode(e) == t for e, t in zip(encoded, heldout, strict=True))
    stats.update(
        heldout_docs=len(heldout),
        heldout_bytes=n_bytes,
        heldout_tokens=n_tokens,
        bytes_per_token=round(n_bytes / n_tokens, 3),
        chars_per_token=round(n_chars / n_tokens, 3),
        roundtrip_ok=roundtrip_ok,
        encode_mb_per_s_one_process=round(n_bytes / 1e6 / encode_seconds, 2),
        encode_tokens_per_s_one_process=round(n_tokens / encode_seconds),
    )
    print(
        f"[4/4] held-out: {n_bytes / 1e6:.1f} MB -> {n_tokens:,} tokens, "
        f"{stats['bytes_per_token']} bytes/token, {stats['chars_per_token']} chars/token, "
        f"round-trip {'OK' if roundtrip_ok else 'FAILED'}, "
        f"encode {stats['encode_mb_per_s_one_process']} MB/s "
        f"({stats['encode_tokens_per_s_one_process']:,} tokens/s) in one process"
    )
    stats_path = args.out.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"stats -> {stats_path}")
    if not roundtrip_ok:
        raise SystemExit("round-trip failed on held-out text")


if __name__ == "__main__":
    main()
