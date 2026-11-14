"""Tests for the strict YAML <-> dataclass config layer (vulpes.config)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from vulpes.config import ConfigError, from_dict, load_config, parse_override, save_config, to_dict


@dataclass
class Inner:
    beta1: float
    beta2: float = 0.95


@dataclass
class Example:
    name: str
    steps: int
    lr: float = 3e-4
    compile: bool = True
    resume_from: str | None = None
    prompts: list[str] = field(default_factory=list)
    adam: Inner = field(default_factory=lambda: Inner(beta1=0.9))


def test_valid_dict_builds_dataclass():
    cfg = from_dict(Example, {"name": "run", "steps": 10, "adam": {"beta1": 0.8}})
    assert cfg == Example(name="run", steps=10, adam=Inner(beta1=0.8))


def test_unknown_key_is_rejected():
    # A typo must fail loudly instead of silently falling back to a default.
    with pytest.raises(ConfigError, match="unknown key.*stpes"):
        from_dict(Example, {"name": "run", "stpes": 10, "steps": 1})


def test_unknown_nested_key_reports_full_path():
    with pytest.raises(ConfigError, match=r"adam\.beta3"):
        from_dict(Example, {"name": "run", "steps": 1, "adam": {"beta1": 0.9, "beta3": 1.0}})


def test_missing_required_key_is_rejected():
    with pytest.raises(ConfigError, match="missing required key: steps"):
        from_dict(Example, {"name": "run"})


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("steps", 1.5),
        ("steps", True),
        ("lr", "fast"),
        ("lr", False),
        ("compile", 1),
        ("name", 3),
        ("prompts", "x"),
    ],
)
def test_wrong_types_are_rejected(key, value):
    with pytest.raises(ConfigError):
        from_dict(Example, {"name": "run", "steps": 1, key: value})


def test_int_is_accepted_as_float():
    # YAML writes `lr: 1` as an int; that must still be a valid float.
    cfg = from_dict(Example, {"name": "run", "steps": 1, "lr": 1})
    assert cfg.lr == 1.0 and isinstance(cfg.lr, float)


def test_optional_accepts_none_and_value():
    assert from_dict(Example, {"name": "r", "steps": 1, "resume_from": None}).resume_from is None
    assert from_dict(Example, {"name": "r", "steps": 1, "resume_from": "ckpt"}).resume_from == "ckpt"


def test_yaml_round_trip(tmp_path):
    cfg = Example(name="run", steps=7, lr=1e-3, prompts=["Once upon a time"], adam=Inner(beta1=0.85))
    path = tmp_path / "nested" / "resolved.yaml"
    save_config(cfg, path)
    assert load_config(Example, path) == cfg
    assert to_dict(cfg)["adam"] == {"beta1": 0.85, "beta2": 0.95}


def test_overrides_take_precedence(tmp_path):
    path = tmp_path / "c.yaml"
    save_config(Example(name="run", steps=7), path)
    assert load_config(Example, path, overrides={"steps": 99}).steps == 99


def test_error_message_includes_file_path(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("name: run\nsteps: 1\nlearning_rate: 0.1\n")
    with pytest.raises(ConfigError, match="bad.yaml"):
        load_config(Example, path)


def test_non_mapping_yaml_is_rejected(tmp_path):
    path = tmp_path / "list.yaml"
    path.write_text("- 1\n- 2\n")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(Example, path)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("lr=3e-4", ("lr", 3e-4)),
        ("compile=false", ("compile", False)),
        ("name=foo", ("name", "foo")),
        ("steps=5", ("steps", 5)),
    ],
)
def test_parse_override(text, expected):
    assert parse_override(text) == expected


def test_parse_override_requires_equals():
    with pytest.raises(ConfigError):
        parse_override("lr")


def test_yaml_scientific_notation_is_float(tmp_path):
    # Plain PyYAML (YAML 1.1) would load `3e-4` as a string; our loader must not.
    path = tmp_path / "c.yaml"
    path.write_text("name: run\nsteps: 1\nlr: 3e-4\n")
    assert load_config(Example, path).lr == pytest.approx(3e-4)


def test_yaml_plain_ints_stay_ints(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("name: run\nsteps: 1000\n")
    assert load_config(Example, path).steps == 1000
