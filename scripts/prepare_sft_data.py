"""Download and convert the fine-tuning conversations (Milestone 10).

Writes ``data/sft/train.jsonl``, ``data/sft/val.jsonl`` and ``data/sft/stats.json``
(format and filtering rules: ``aupus/data/sft.py``). Downloads ~80 MB at pinned
revisions:

| source   | dataset (revision)                            | license      | kept (measured)        |
|----------|-----------------------------------------------|--------------|------------------------|
| oasst2   | OpenAssistant/oasst2 (179dd21), train + valid | Apache-2.0   | 5,270 English conversations |
| dolly    | databricks/databricks-dolly-15k (bdd27f4)     | CC BY-SA 3.0 | 15,011                 |
| identity | data/identity.jsonl (hand-written, in git)    | Apache-2.0   | whatever you wrote     |

The identity set is tiny next to the rest, so its conversations are repeated
``--identity-repeat`` times in the train split (none go to val). Leave a source
out with ``--sources``, e.g. ``--sources oasst2,identity`` to train without Dolly.

Usage:
    python scripts/prepare_sft_data.py
    python scripts/prepare_sft_data.py --sources oasst2,identity
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from aupus.data.sft import (
    dolly_conversations,
    identity_conversations,
    oasst_conversations,
    split,
    write_jsonl,
)

OASST2 = ("OpenAssistant/oasst2", "179dd21fc55192153d94adb0e0ce8f69e222bf75")
OASST2_FILES = (
    "data/train-00000-of-00001-88ba0162028a73fc.parquet",
    "data/validation-00000-of-00001-1deeef95c3248fe0.parquet",
)
DOLLY = ("databricks/databricks-dolly-15k", "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a")
DOLLY_FILE = "databricks-dolly-15k.jsonl"
SOURCES = ("oasst2", "dolly", "identity")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/sft"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--identity", type=Path, default=Path("data/identity.jsonl"))
    parser.add_argument("--identity-repeat", type=int, default=3)
    parser.add_argument("--sources", default=",".join(SOURCES), help=f"comma-separated, from {SOURCES}")
    parser.add_argument("--val-fraction", type=float, default=0.05)
    args = parser.parse_args()

    sources = [s for s in args.sources.split(",") if s]
    if set(sources) - set(SOURCES):
        parser.error(f"unknown source(s): {sorted(set(sources) - set(SOURCES))}")
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    def fetch(repo: tuple[str, str], filename: str) -> str:
        name, revision = repo
        local = args.raw_dir / name.split("/")[1]
        return hf_hub_download(name, filename, repo_type="dataset", revision=revision, local_dir=local)

    stats: dict[str, dict] = {}
    conversations = []
    if "oasst2" in sources:
        rows = [r for f in OASST2_FILES for r in pq.read_table(fetch(OASST2, f)).to_pylist()]
        counts: Counter = Counter()
        conversations += oasst_conversations(rows, stats=counts)
        stats["oasst2"] = dict(counts)
    if "dolly" in sources:
        with open(fetch(DOLLY, DOLLY_FILE), encoding="utf-8") as f:
            counts = Counter()
            conversations += dolly_conversations((json.loads(line) for line in f), stats=counts)
        stats["dolly"] = dict(counts)
    train, val = split(conversations, args.val_fraction)

    if "identity" in sources:
        if args.identity.exists():
            identity = identity_conversations(args.identity)
            train += identity * args.identity_repeat
            stats["identity"] = {"kept": len(identity), "repeat": args.identity_repeat}
        else:
            print(f"warning: {args.identity} not found; training without identity conversations")

    n_train = write_jsonl(train, args.out_dir / "train.jsonl")
    n_val = write_jsonl(val, args.out_dir / "val.jsonl")
    by_source = {
        name: dict(Counter(c["source"] for c in convs)) for name, convs in (("train", train), ("val", val))
    }
    summary = {"sources": stats, "splits": by_source, "val_fraction": args.val_fraction}
    (args.out_dir / "stats.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {n_train:,} train / {n_val:,} val conversations to {args.out_dir}")


if __name__ == "__main__":
    main()
