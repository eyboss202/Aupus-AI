"""Download the evaluation benchmarks and convert them to one JSON-lines format (Milestone 9).

Writes ``data/eval/<task>.jsonl``, one ``{"id", "context", "choices", "label"}``
object per line (see ``vulpes/eval/tasks.py`` for the prompt formats). Every file
is fetched at a pinned dataset revision, so the questions never change under us.
About 7 MB in total.

These are test sets: they are only ever used to *measure* models, never to train them.

| task      | dataset (revision)                        | split      | n      | license       |
|-----------|-------------------------------------------|------------|--------|---------------|
| hellaswag | Rowan/hellaswag (218ec52)                 | validation | 10,042 | MIT           |
| arc_easy  | allenai/ai2_arc, ARC-Easy (210d026)       | test       | 2,376  | CC BY-SA 4.0  |
| piqa      | baber/piqa (142f6d7), parquet copy of PIQA | validation | 1,838  | AFL-3.0       |

(The original ``ybisk/piqa`` repo only has a loading script, which current
``datasets`` versions no longer run; lm-evaluation-harness uses the same parquet copy.)

Usage:
    python scripts/download_eval_data.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vulpes.eval import CONVERTERS, write_examples

SOURCES = {
    "hellaswag": (
        "Rowan/hellaswag",
        "218ec52e09a7e7462a5400043bb9a69a41d06b76",
        "data/validation-00000-of-00001.parquet",
    ),
    "arc_easy": (
        "allenai/ai2_arc",
        "210d026faf9955653af8916fad021475a3f00453",
        "ARC-Easy/test-00000-of-00001.parquet",
    ),
    "piqa": ("baber/piqa", "142f6d7367fd9877f0fb3b5734ea6a545f54cdd1", "piqa_validation.parquet"),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/eval"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    for task, (repo, revision, filename) in SOURCES.items():
        path = hf_hub_download(
            repo,
            filename,
            repo_type="dataset",
            revision=revision,
            local_dir=args.raw_dir / repo.split("/")[1],
        )
        records = pq.read_table(path).to_pylist()
        n = write_examples((CONVERTERS[task](r) for r in records), args.out_dir / f"{task}.jsonl")
        print(f"{task}: {n:,} examples -> {args.out_dir / task}.jsonl")


if __name__ == "__main__":
    main()
