"""Tests for model presets and the analytic parameter / FLOP accounting."""

from __future__ import annotations

import dataclasses

import pytest

from aupus.config import ConfigError
from aupus.model.config import PRESET_DIR, ModelConfig, flops_per_token, param_count

# Target sizes from the spec (docs/PLAN.md), with some tolerance.
EXPECTED_TOTAL_PARAMS = {
    "debug": (3e6, 8e6),
    "tiny": (25e6, 40e6),
    "small": (110e6, 140e6),
    "base": (300e6, 400e6),
}


def test_every_preset_file_is_covered():
    assert {p.stem for p in PRESET_DIR.glob("*.yaml")} == set(EXPECTED_TOTAL_PARAMS)


@pytest.mark.parametrize("name", sorted(EXPECTED_TOTAL_PARAMS))
def test_preset_loads_and_has_expected_size(name):
    cfg = ModelConfig.preset(name)
    lo, hi = EXPECTED_TOTAL_PARAMS[name]
    assert lo <= param_count(cfg)["total"] <= hi


def test_small_preset_exact_count():
    # Pinned so any accidental change to the Aupus 0.1 shape is noticed in review.
    assert param_count(ModelConfig.preset("small"))["total"] == 125_264_640


def test_unknown_preset_lists_available():
    with pytest.raises(ConfigError, match="available: base, debug, small, tiny"):
        ModelConfig.preset("huge")


def test_param_count_components_add_up():
    cfg = ModelConfig.preset("tiny")
    p = param_count(cfg)
    assert p["total"] == p["embedding"] + p["non_embedding"] + p["lm_head_untied"]
    assert p["lm_head_untied"] == 0  # tied by default


def test_untied_head_adds_vocab_times_width():
    cfg = ModelConfig.preset("debug")
    untied = dataclasses.replace(cfg, tie_embeddings=False)
    assert param_count(untied)["total"] - param_count(cfg)["total"] == cfg.vocab_size * cfg.d_model


def test_gqa_reduces_params_vs_mha():
    cfg = ModelConfig.preset("small")
    mha = dataclasses.replace(cfg, n_kv_heads=cfg.n_heads)
    assert param_count(mha)["total"] > param_count(cfg)["total"]


@pytest.mark.parametrize(
    "changes",
    [
        {"d_model": 100, "n_heads": 3},  # d_model not divisible by heads
        {"n_heads": 8, "n_kv_heads": 3},  # heads not a multiple of kv heads
        {"d_model": 36, "n_heads": 4, "n_kv_heads": 2},  # head_dim 9 is odd -> RoPE impossible
        {"n_layers": 0},
    ],
)
def test_invalid_shapes_are_rejected(changes):
    base = dataclasses.asdict(ModelConfig.preset("debug"))
    with pytest.raises(ConfigError):
        ModelConfig(**{**base, **changes})


def test_flops_accounting():
    cfg = ModelConfig.preset("small")
    f = flops_per_token(cfg)
    assert f["train"] == 3 * f["forward"]
    # Training must cost at least the classic 6*N lower bound (N = non-embedding params).
    assert f["train"] > 6 * param_count(cfg)["non_embedding"]
    # Longer context -> more attention FLOPs.
    assert flops_per_token(cfg, 2048)["train"] > flops_per_token(cfg, 512)["train"]
