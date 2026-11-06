"""Pretrain an Aupus model (Milestones 6 and 8).

Start a run from a config in ``configs/train/``, optionally overriding any key
(dotted keys reach nested settings; values are parsed as YAML)::

    python scripts/train.py --config configs/train/tiny.yaml
    python scripts/train.py --config configs/train/tiny.yaml max_steps=200 optim.lr=1e-3 run_name=tiny-lr1e3

Continue a stopped run (Ctrl-C, crash, preempted cloud box) from its newest
checkpoint. The run's own saved ``config.yaml`` is used; overrides still apply
(e.g. a longer ``max_steps``)::

    python scripts/train.py --resume runs/tiny

Everything a run writes goes to ``<out_dir>/<run_name>/``: ``config.yaml``,
``model.yaml``, ``train.log``, ``metrics.jsonl``, ``tb/`` (TensorBoard:
``tensorboard --logdir runs``), ``step_NNNNNN.pt`` checkpoints and ``best.pt``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from aupus.config import load_yaml, parse_override
from aupus.train import TrainConfig, Trainer


def main() -> None:
    fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=fmt)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=Path, help="run config YAML (starts a new run)")
    group.add_argument("--resume", type=Path, metavar="RUN_DIR", help="continue the run in this directory")
    parser.add_argument("overrides", nargs="*", metavar="key=value", help="config overrides")
    args = parser.parse_args()

    overrides = dict(parse_override(o) for o in args.overrides)
    if args.resume:
        saved = args.resume / "config.yaml"
        if not saved.exists():
            raise SystemExit(f"{saved} not found: is {args.resume} a run directory?")
        # The run directory is wherever it is now (it may have been copied off a cloud box).
        overrides = {"out_dir": str(args.resume.parent), "run_name": args.resume.name, **overrides}
        cfg = TrainConfig.from_dict(load_yaml(saved), overrides, source=str(saved))
    else:
        cfg = TrainConfig.from_yaml(args.config, overrides)

    def log(msg: str) -> None:
        print(msg, flush=True)

    trainer = Trainer(cfg, resume=args.resume is not None, log=log)
    summary = trainer.train()
    log(json.dumps(summary))
    if not summary["finished"]:
        log(f"stopped early; continue with: python scripts/train.py --resume {cfg.run_dir.as_posix()}")
        sys.exit(130)


if __name__ == "__main__":
    main()
