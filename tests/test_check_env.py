"""Smoke test: the environment report runs on any machine (including CI)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_env.py"


def _load_check_env():
    spec = importlib.util.spec_from_file_location("check_env", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_collect_returns_core_fields():
    check_env = _load_check_env()
    report = check_env.collect()
    for key in ("os", "python", "cpu", "ram", "disk_project_drive", "torch"):
        assert key in report
    assert report["torch"]["installed"] is True
    assert report["disk_project_drive"]["free_gib"] >= 0
    assert isinstance(check_env.verdict(report), list)
