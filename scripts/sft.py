"""Fine-tune a Base checkpoint on conversations: Vulpes 0.1 Base -> Vulpes 0.1 Instruct (Milestone 10).

Needs the conversations from ``scripts/prepare_sft_data.py`` and a Base
checkpoint (``init_from`` in the config, usually ``runs/<base run>/best.pt``)::

    python scripts/sft.py --config configs/sft/vulpes-0.1-instruct.yaml
    python scripts/sft.py --config configs/sft/vulpes-0.1-instruct.yaml init_from=runs/tiny/best.pt run_name=t
    python scripts/sft.py --resume runs/vulpes-0.1-instruct        # after Ctrl-C or a crash

Same run layout, logging and Ctrl-C behaviour as ``scripts/train.py``. The run
stops early when validation loss stops improving; ``best.pt`` is the model to
keep (export it with ``scripts/export_release.py``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vulpes.config import load_yaml, parse_override
from vulpes.train.sft import SFTConfig, SFTTrainer


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=Path, help="SFT config YAML (starts a new run)")
    group.add_argument("--resume", type=Path, metavar="RUN_DIR", help="continue the run in this directory")
    parser.add_argument("overrides", nargs="*", metavar="key=value", help="config overrides")
    args = parser.parse_args()

    overrides = dict(parse_override(o) for o in args.overrides)
    if args.resume:
        saved = args.resume / "config.yaml"
        if not saved.exists():
            raise SystemExit(f"{saved} not found: is {args.resume} a run directory?")
        overrides = {"out_dir": str(args.resume.parent), "run_name": args.resume.name, **overrides}
        cfg = SFTConfig.from_dict(load_yaml(saved), overrides, source=str(saved))
    else:
        cfg = SFTConfig.from_yaml(args.config, overrides)

    def log(msg: str) -> None:
        print(msg, flush=True)

    trainer = SFTTrainer(cfg, resume=args.resume is not None, log=log)
    summary = trainer.train()
    log(json.dumps(summary))
    if not summary["finished"]:
        log(f"stopped early; continue with: python scripts/sft.py --resume {cfg.run_dir.as_posix()}")
        sys.exit(130)


if __name__ == "__main__":
    main()
