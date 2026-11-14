"""Show how far a training run is, and what it is using on this PC (safe to run while it trains).

It only reads the run directory (``config.yaml``, ``metrics.jsonl``, checkpoint files) and asks
``nvidia-smi`` about the GPU; it never touches the training process.

Usage:
    python scripts/train_status.py                          # runs/vulpes-0.1-base
    python scripts/train_status.py --watch 30        # refresh every 30 s (Ctrl-C quits this, not training)
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

from vulpes.train.checkpoint import list_checkpoints


def _duration(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m = rem // 60
    return f"{d} d {h} h" if d else f"{h} h {m} min" if h else f"{m} min"


def read_status(run: Path, window: int = 20) -> dict[str, Any]:
    """Progress, losses, speed and checkpoints of a run, from its files."""
    cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    max_steps = cfg["max_steps"]
    tokens_per_step = cfg["micro_batch"] * cfg["grad_accum"] * cfg["seq_len"]
    train, val = [], []
    metrics = run / "metrics.jsonl"
    if metrics.exists():
        for line in metrics.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if "train/loss" in record:
                train.append(record)
            if "val/loss" in record:
                val.append(record)
    step = train[-1]["step"] if train else 0
    recent = [r["perf/tokens_per_sec"] for r in train[-window:]]
    tok_s = sum(recent) / len(recent) if recent else None
    best = min(val, key=lambda r: r["val/loss"]) if val else None
    checkpoints = list_checkpoints(run)
    last_ckpt = checkpoints[-1] if checkpoints else None
    return {
        "run": run,
        "step": step,
        "max_steps": max_steps,
        "percent": 100 * step / max_steps,
        "tokens": step * tokens_per_step,
        "total_tokens": max_steps * tokens_per_step,
        "tok_s": tok_s,
        "seconds_per_step": tokens_per_step / tok_s if tok_s else None,
        "eta_seconds": (max_steps - step) * tokens_per_step / tok_s if tok_s else None,
        "train_loss": train[-1]["train/loss"] if train else None,
        "first_val": val[0]["val/loss"] if val else None,
        "best_val": (best["val/loss"], best["step"]) if best else None,
        "checkpoint": (last_ckpt[0], time.time() - last_ckpt[1].stat().st_mtime) if last_ckpt else None,
        "peak_mem_gib": train[-1].get("perf/peak_mem_gib") if train else None,
    }


def gpu_status() -> str | None:
    """One line from nvidia-smi, or None without an NVIDIA GPU."""
    if not shutil.which("nvidia-smi"):
        return None
    query = "temperature.gpu,power.draw,utilization.gpu,memory.used,memory.total"
    try:
        out = (
            subprocess.run(
                ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
            .stdout.strip()
            .splitlines()[0]
        )
    except (OSError, subprocess.SubprocessError, IndexError):
        return None
    temp, power, util, used, total = (v.strip() for v in out.split(","))
    memory = f"{float(used) / 1024:.1f} / {float(total) / 1024:.1f} GB memory"
    return f"{temp} °C · {float(power):.0f} W · {util}% busy · {memory}"


def render(s: dict[str, Any], gpu: str | None) -> str:
    width = 30
    filled = round(width * s["percent"] / 100)
    lines = [
        f"Vulpes training: {s['run'].as_posix()}",
        f"  progress    step {s['step']:,} / {s['max_steps']:,}  ({s['percent']:.1f}%)  "
        f"[{'#' * filled}{'.' * (width - filled)}]",
        f"  tokens      {s['tokens'] / 1e6:,.1f}M / {s['total_tokens'] / 1e9:.2f}B",
    ]
    if s["tok_s"]:
        lines.append(f"  speed       {s['tok_s']:,.0f} tok/s, ~{s['seconds_per_step']:.0f} s per step")
        lines.append(f"  time left   ~{_duration(s['eta_seconds'])} of training (stops not counted)")
    else:
        lines.append("  speed       no step finished yet")
    if s["train_loss"] is not None:
        lines.append(f"  train loss  {s['train_loss']:.3f}")
    if s["best_val"]:
        best, at = s["best_val"]
        lines.append(f"  val loss    best {best:.3f} at step {at:,} (started at {s['first_val']:.2f})")
    if s["checkpoint"]:
        step, age = s["checkpoint"]
        lines.append(f"  checkpoint  step {step:,}, saved {_duration(age)} ago")
    else:
        lines.append("  checkpoint  none yet (the first one is written at step 10)")
    if gpu:
        lines.append(f"  GPU now     {gpu}")
    if s["peak_mem_gib"]:
        lines.append(f"  training    peak {s['peak_mem_gib']:.2f} GB of GPU memory in tensors")
    return "\n".join(lines)


def main() -> None:
    fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=fmt)
    parser.add_argument("--run", type=Path, default=Path("runs/vulpes-0.1-base"))
    parser.add_argument("--watch", type=int, default=0, metavar="SECONDS", help="refresh every N seconds")
    args = parser.parse_args()
    if not (args.run / "config.yaml").exists():
        raise SystemExit(
            f"{args.run} is not a run directory (no config.yaml); run this from the Aupus-AI folder"
        )
    while True:
        text = render(read_status(args.run), gpu_status())
        if not args.watch:
            print(text)
            return
        print("\033[2J\033[H" + text + f"\n\n  (refreshing every {args.watch} s, Ctrl-C to quit)", flush=True)
        try:
            time.sleep(args.watch)
        except KeyboardInterrupt:
            return


if __name__ == "__main__":
    main()
