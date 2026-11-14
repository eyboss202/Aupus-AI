"""Stream a small FineWeb-Edu sample for tokenizer training (Milestone 3).

The tokenizer only needs a representative slice of the pretraining
distribution, not all of it: ~300 MB of text gives stable merge statistics for
a 32k vocab. Documents are streamed (nothing but the sample is kept on disk),
NFC-normalised like the real data pipeline will do, and written as JSON lines
``{"text": ...}``.

Two files are written:
  * ``<out-dir>/train.jsonl``: the first ``--train-mb`` MB, used to train BPE;
  * ``<out-dir>/heldout.jsonl``: the next ``--heldout-mb`` MB, never seen in
    training, used to measure compression honestly.

Usage:
    python scripts/download_tokenizer_sample.py                # 300 MB + 10 MB
    python scripts/download_tokenizer_sample.py --train-mb 20  # quick test
"""

from __future__ import annotations

import argparse
import json
import time
import unicodedata
from pathlib import Path

DATASET = "HuggingFaceFW/fineweb-edu"
SUBSET = "sample-10BT"  # ODC-By 1.0; see docs/PLAN.md section 5
# Pinned dataset revision: the stream order (and so the exact sample) is reproducible.
# This is the revision data/tokenizer/vulpes-bpe-32k.json was trained from.
REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"


def main() -> None:
    fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=fmt)
    parser.add_argument("--out-dir", type=Path, default=Path("data/tokenizer_sample"))
    parser.add_argument("--train-mb", type=float, default=300.0)
    parser.add_argument("--heldout-mb", type=float, default=10.0)
    args = parser.parse_args()

    from datasets import load_dataset  # optional dependency: pip install -e ".[data]"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stream = iter(load_dataset(DATASET, name=SUBSET, split="train", streaming=True, revision=REVISION))
    start = time.time()
    for name, limit_mb in (("train", args.train_mb), ("heldout", args.heldout_mb)):
        limit = int(limit_mb * 1e6)
        written = docs = 0
        path = args.out_dir / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            while written < limit:
                text = unicodedata.normalize("NFC", next(stream)["text"])
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                written += len(text.encode("utf-8"))
                docs += 1
        print(f"{path}: {docs:,} docs, {written / 1e6:.1f} MB of text ({time.time() - start:.0f}s elapsed)")


if __name__ == "__main__":
    main()
