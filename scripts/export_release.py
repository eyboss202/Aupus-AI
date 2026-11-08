"""Export a training checkpoint as a weights-only release directory (Milestone 11).

    python scripts/export_release.py --checkpoint runs/aupus-0.1-instruct/best.pt \\
        --out release/aupus-0.1-instruct
    python scripts/export_release.py --checkpoint runs/aupus-0.1-base/best.pt \\
        --out release/aupus-0.1-base --dtype bf16

The directory holds ``model.safetensors``, ``model.yaml``, ``tokenizer.json`` and
``release.json`` (format: ``aupus/model/io.py``), plus ``LICENSE`` and
``MODEL_CARD.md`` copied from the repository. Zip it and attach it to a GitHub
Release; weights never go into git. ``scripts/chat.py --model <dir>`` and
``scripts/evaluate.py --model <dir>`` load it directly.

Before publishing, check the export: it is loaded back and its logits are
compared with the checkpoint's on a fixed input (exact for fp32; for fp16/bf16
the difference is reported).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

from aupus.model.io import export_release, load_model

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="training checkpoint (.pt)")
    parser.add_argument("--out", type=Path, required=True, help="release directory to create")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--force", action="store_true", help="replace an existing, non-empty --out")
    args = parser.parse_args()

    out = export_release(args.checkpoint, args.out, dtype=args.dtype, overwrite=args.force)
    for name in ("LICENSE", "MODEL_CARD.md"):
        if (ROOT / name).exists():
            shutil.copyfile(ROOT / name, out / name)

    original, exported = load_model(args.checkpoint), load_model(out)
    x = torch.arange(min(64, original.model.cfg.context_length))[None] % original.model.cfg.vocab_size
    with torch.no_grad():
        diff = (original.model(x).logits - exported.model(x).logits).abs().max().item()
    if args.dtype == "fp32" and diff != 0:
        raise SystemExit(f"export check failed: logits differ by {diff}")
    size = sum(p.stat().st_size for p in out.iterdir()) / 1e6
    print(f"wrote {out} ({size:.1f} MB); max logit difference vs checkpoint: {diff:.2e}")


if __name__ == "__main__":
    main()
