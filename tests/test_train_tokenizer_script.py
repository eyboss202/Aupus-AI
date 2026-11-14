"""Smoke test: scripts/train_tokenizer.py runs end to end on a tiny JSONL corpus."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from vulpes.tokenizer import Tokenizer

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_tokenizer.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("train_tokenizer", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_train_tokenizer_script(tmp_path, monkeypatch):
    docs = [f"Document {i}: the cat sat on the mat, and the dog sat on the log." for i in range(50)]
    for name, texts in (("train", docs), ("heldout", docs[:5])):
        (tmp_path / f"{name}.jsonl").write_text(
            "".join(json.dumps({"text": t}) + "\n" for t in texts), encoding="utf-8"
        )
    out = tmp_path / "tok"
    argv = ["train_tokenizer.py", "--input", str(tmp_path / "train.jsonl"), "--heldout"]
    argv += [str(tmp_path / "heldout.jsonl"), "--out", str(out), "--vocab-size", "280", "--workers", "1"]
    monkeypatch.setattr(sys, "argv", argv)
    _load_script().main()

    tok = Tokenizer.load(out.with_suffix(".json"))
    assert tok.vocab_size == 280
    stats = json.loads(out.with_suffix(".stats.json").read_text(encoding="utf-8"))
    assert stats["roundtrip_ok"] is True
    assert stats["bytes_per_token"] > 1.0
    assert out.with_suffix(".vocab.txt").exists()
