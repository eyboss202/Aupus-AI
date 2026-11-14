"""Tests for scripts/check_identity.py (the identity-file checker in docs/IDENTITY.md)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from vulpes.tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def checker():
    spec = importlib.util.spec_from_file_location("check_identity", ROOT / "scripts" / "check_identity.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tok():
    return Tokenizer.load(ROOT / "data" / "tokenizer" / "vulpes-bpe-32k.json")


def write(path, *records):
    lines = [r if isinstance(r, str) else json.dumps(r) for r in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def conv(*turns):
    roles = ["user", "assistant"] * len(turns)
    return {"messages": [{"role": r, "content": t} for r, t in zip(roles, turns, strict=False)]}


def test_good_file_has_no_errors(checker, tok, tmp_path):
    records = [conv(f"Question number {i} about you?", f"Answer {i}: I'm Vulpes.") for i in range(40)]
    records.append(conv("What is your name?", "I'm Vulpes.", "And who made you?", "An open-source project."))
    errors, warnings, stats = checker.check(write(tmp_path / "id.jsonl", *records), tok, 1024)
    assert errors == [] and warnings == []
    assert stats["conversations"] == 41 and stats["multi_turn"] == 1


def test_errors_point_at_the_line(checker, tok, tmp_path):
    path = write(tmp_path / "id.jsonl", conv("Hi?", "Hello."), "{not json", {"text": "x"})
    errors, _, _ = checker.check(path, tok, 1024)
    assert errors[0].startswith("line 2: not valid JSON") and errors[1].startswith("line 3: needs")
    path = write(
        tmp_path / "id.jsonl", conv("Hi?", "Hello."), {"messages": [{"role": "user", "content": "x"}]}
    )
    (error,) = checker.check(path, tok, 1024)[0]
    assert "id.jsonl:2" in error and "assistant" in error
    errors, _, _ = checker.check(write(tmp_path / "id.jsonl", conv("Hi?", "word " * 2000)), tok, 1024)
    assert "longer than the 1024-token context window" in errors[0]


def test_warnings(checker, tok, tmp_path):
    path = write(
        tmp_path / "id.jsonl",
        conv("What's your name?", "I'm a small model."),  # name question, name never said
        conv("What's your name?", "I'm Vulpes."),  # same question twice
        conv("Contact?", "Write to me at someone@example.com or https://example.com"),
        conv("Como te chamas?", "Chamo-me Vulpes e não falo bem português, lá está."),
    )
    _, warnings, _ = checker.check(path, tok, 1024)
    text = "\n".join(warnings)
    assert "no reply says 'Vulpes'" in text and "appears 2 times" in text
    assert "an email address" in text and "a link" in text
    assert "doesn't look like English" in text and "only 4 conversations" in text
