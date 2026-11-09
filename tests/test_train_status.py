"""Tests for scripts/train_status.py (reads a run directory; never touches the training process)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_script():
    spec = importlib.util.spec_from_file_location("train_status", ROOT / "scripts" / "train_status.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_status_from_run_files(tmp_path):
    status = load_script()
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.yaml").write_text(
        yaml.safe_dump({"max_steps": 100, "micro_batch": 1, "grad_accum": 4, "seq_len": 256})
    )
    records = [{"step": 0, "val/loss": 10.5}]
    records += [{"step": s, "train/loss": 9 - s / 10, "perf/tokens_per_sec": 1000.0} for s in range(1, 26)]
    records += [{"step": 20, "val/loss": 7.0}, {"step": 25, "val/loss": 7.2}]
    (run / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    (run / "step_000020.pt").write_bytes(b"")

    s = status.read_status(run)
    assert (s["step"], s["percent"], s["tokens"]) == (25, 25.0, 25 * 1024)
    assert s["best_val"] == (7.0, 20) and s["first_val"] == 10.5
    assert s["eta_seconds"] == 75 * 1024 / 1000 and s["checkpoint"][0] == 20
    text = status.render(s, "60 °C · 90 W · 100% busy · 2.4 / 4.0 GB memory")
    assert "step 25 / 100  (25.0%)" in text and "best 7.000 at step 20" in text and "60 °C" in text


def test_status_before_the_first_step(tmp_path):
    status = load_script()
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.yaml").write_text(
        yaml.safe_dump({"max_steps": 10, "micro_batch": 1, "grad_accum": 1, "seq_len": 8})
    )
    text = status.render(status.read_status(run), None)
    assert "step 0 / 10" in text and "no step finished yet" in text and "none yet" in text
