"""Evaluate a model: zero-shot HellaSwag / ARC-Easy / PIQA and validation perplexity (Milestone 9).

The model can be a training checkpoint (``runs/<name>/best.pt``) or a release
directory (``scripts/export_release.py``). Benchmarks come from
``scripts/download_eval_data.py``. Results are printed and written to JSON.

Usage:
    python scripts/download_eval_data.py                                   # once, ~7 MB
    python scripts/evaluate.py --model runs/vulpes-0.1-base/best.pt --val-data data/pretrain/fineweb_edu
    python scripts/evaluate.py --model runs/tiny/best.pt --tasks piqa --limit 200   # quick look

Cost: the three tasks are ~14K questions, ~45K scored continuations, ~3.8M
tokens of forward passes (no backward), so a few minutes on a GPU and roughly
10-40 min on a CPU depending on the model size. ``--limit`` takes the first N
questions of each task; the standard error printed next to each accuracy shows
how much that costs in precision.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from vulpes.eval import RANDOM_BASELINE, TASKS, evaluate_loss, read_examples, run_task
from vulpes.model.io import load_model
from vulpes.train import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", type=Path, required=True, help="checkpoint .pt or release directory")
    parser.add_argument("--tasks", default=",".join(TASKS), help=f"comma-separated, from {', '.join(TASKS)}")
    parser.add_argument("--eval-dir", type=Path, default=Path("data/eval"))
    parser.add_argument("--val-data", type=Path, help="prepared dataset dir: also report val perplexity")
    parser.add_argument("--val-batches", type=int, default=None, help="limit val perplexity to N batches")
    parser.add_argument("--limit", type=int, default=None, help="first N questions per task")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--out", type=Path, help="results JSON (default: next to the model)")
    args = parser.parse_args()

    tasks = [t for t in args.tasks.split(",") if t]
    unknown = sorted(set(tasks) - set(TASKS))
    if unknown:
        parser.error(f"unknown task(s): {', '.join(unknown)}")
    device = resolve_device(args.device)
    loaded = load_model(args.model, device=device)
    model, tok = loaded.model, loaded.tokenizer
    params = model.num_params() / 1e6
    print(f"model {args.model} | {params:.1f}M params | step {loaded.info.get('step')} | {device}")

    results: dict = {"model": args.model.as_posix(), "info": loaded.info, "tasks": {}}
    if args.val_data:
        t0 = time.perf_counter()
        val = evaluate_loss(model, tok, args.val_data, max_batches=args.val_batches)
        results["val"] = {**val, "data_dir": args.val_data.as_posix()}
        print(
            f"val: loss {val['loss']:.4f} | ppl {val['ppl']:.2f} | {val['bits_per_byte']:.4f} bits/byte "
            f"| {val['tokens'] / 1e6:.2f}M tokens | {time.perf_counter() - t0:.0f} s"
        )

    print(f"\n{'task':<10} {'n':>6} {'acc':>15} {'acc_norm':>9} {'random':>7} {'time':>6}")
    for task in tasks:
        path = args.eval_dir / f"{task}.jsonl"
        if not path.exists():
            raise SystemExit(f"{path} not found: run scripts/download_eval_data.py first")
        examples = list(read_examples(path))[: args.limit]
        t0 = time.perf_counter()
        r = run_task(task, model, tok, examples, batch_size=args.batch_size, progress=True)
        seconds = time.perf_counter() - t0
        results["tasks"][task] = {
            "n": r.n,
            "acc": r.acc,
            "acc_stderr": r.acc_stderr,
            "acc_norm": r.acc_norm,
            "random": RANDOM_BASELINE[task],
            "seconds": seconds,
        }
        print(
            f"{task:<10} {r.n:>6} {r.acc:>8.2%} ±{r.acc_stderr:5.2%} {r.acc_norm:>9.2%} "
            f"{RANDOM_BASELINE[task]:>7.0%} {seconds:>5.0f}s"
        )

    out = args.out or (
        args.model / "eval.json"
        if args.model.is_dir()
        else args.model.with_name(f"eval_{args.model.stem}.json")
    )
    results["torch"] = torch.__version__
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
