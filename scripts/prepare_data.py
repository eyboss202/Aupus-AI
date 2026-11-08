"""Turn raw documents into pretraining token shards (Milestones 5 and 7).

Reads a JSON-lines file (``{"text": ...}`` per line), local parquet files, or
streams FineWeb-Edu directly, then runs normalise -> filter -> dedup -> split -> tokenize -> shard
(see ``aupus/data/pipeline.py``). Writes ``meta.json``, ``stats.json`` and the
``.bin`` shards to ``--out``.

Usage:
    # Milestone 5: the 300 MB sample already on disk
    python scripts/prepare_data.py --input data/tokenizer_sample/train.jsonl --out data/pretrain/sample
    # Milestone 7 (big download, ask first): download the parquet files, then process them
    python scripts/download_fineweb_edu.py --files 4
    python scripts/prepare_data.py --source parquet --input data/raw/fineweb_edu \
        --out data/pretrain/fineweb_edu
    # (or stream instead of downloading; not resumable if the connection drops)
    python scripts/prepare_data.py --source fineweb-edu --max-docs 2800000 --out data/pretrain/fineweb_edu
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tqdm import tqdm

from aupus.config import load_config
from aupus.data import DataConfig, iter_fineweb_edu, iter_jsonl, iter_parquet, prepare


def main() -> None:
    fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=fmt)
    parser.add_argument(
        "--input", type=Path, help="JSON-lines file, or a parquet file / directory (searched recursively)"
    )
    parser.add_argument("--source", choices=["jsonl", "parquet", "fineweb-edu"], default="jsonl")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/data/fineweb_edu.yaml"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer/aupus-bpe-32k.json"))
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--max-docs", type=int, default=None)
    args = parser.parse_args()

    if args.source in ("jsonl", "parquet") and args.input is None:
        parser.error(f"--input is required with --source {args.source}")
    if args.source == "jsonl":
        texts = iter_jsonl(args.input)
    elif args.source == "parquet":
        # Sorted by path, which for FineWeb-Edu (000_00000.parquet, 001_...) is the streaming order.
        paths = sorted(args.input.rglob("*.parquet")) if args.input.is_dir() else [args.input]
        if not paths:
            parser.error(f"no .parquet files under {args.input}")
        print(f"reading {len(paths)} parquet file(s): {', '.join(p.name for p in paths)}")
        texts = iter_parquet(paths)
    else:
        texts = iter_fineweb_edu()
    cfg = load_config(DataConfig, args.config)
    with tqdm(total=args.max_docs, unit="doc", mininterval=2.0) as bar:
        stats = prepare(texts, args.out, args.tokenizer, cfg, args.workers, args.max_docs, progress=bar)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
