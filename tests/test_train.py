"""Tests for the pretraining stack: config, optimizer groups, LR schedule, loop, checkpoints, resume."""

from __future__ import annotations

import importlib.util
import json
import math
import random
import sys
from pathlib import Path

import pytest
import torch
import yaml

from aupus.config import ConfigError
from aupus.data import DataConfig, prepare
from aupus.model import ModelConfig, Transformer
from aupus.tokenizer import Tokenizer
from aupus.train import (
    OptimConfig,
    TrainConfig,
    Trainer,
    lr_at,
    param_groups,
    pretrain,
    resolve_model_config,
    resolve_precision,
)
from aupus.train import checkpoint as ckpt

ROOT = Path(__file__).resolve().parents[1]
WORDS = [
    "cat",
    "dog",
    "sun",
    "tree",
    "river",
    "stone",
    "light",
    "house",
    "water",
    "garden",
    "bird",
    "music",
    "story",
    "bridge",
    "cloud",
    "apple",
]


def prose(rng: random.Random, n: int = 15) -> str:
    sentences = []
    for _ in range(n):
        body = " ".join(rng.choice(WORDS) for _ in range(rng.randint(6, 12)))
        sentences.append(f"The {body} and that is the way to learn with care.")
    return " ".join(sentences)


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    """A tiny tokenizer, a ~60K-token dataset made by the real pipeline, and a tiny model YAML."""
    root = tmp_path_factory.mktemp("train")
    rng = random.Random(3)
    docs = [prose(rng) for _ in range(400)]
    tok_path = root / "tok.json"
    Tokenizer.train(docs[:50], vocab_size=256 + 100 + 7).save(tok_path)
    prepare(docs, root / "data", tok_path, DataConfig(val_fraction=0.1))
    model_yaml = root / "model.yaml"
    model_yaml.write_text(
        yaml.safe_dump(
            {"vocab_size": 363, "d_model": 32, "n_layers": 2, "n_heads": 2, "n_kv_heads": 1, "ffn_hidden": 64}
        )
    )
    return root, tok_path, model_yaml


def make_cfg(setup, tmp_path, **overrides) -> TrainConfig:
    _, tok_path, model_yaml = setup
    base = {
        "run_name": "run",
        "out_dir": str(tmp_path),
        "model": str(model_yaml),
        "data_dir": str(setup[0] / "data"),
        "tokenizer": str(tok_path),
        "seq_len": 32,
        "micro_batch": 4,
        "grad_accum": 2,
        "max_steps": 6,
        "optim": {"lr": 3e-3, "min_lr": 3e-4, "warmup_steps": 2},
        "device": "cpu",
        "compile": False,
        "eval_every": 3,
        "eval_batches": 2,
        "save_every": 3,
        "log_every": 1,
    }
    return TrainConfig.from_dict(base, overrides)


def quiet(_msg: str) -> None:
    pass


# ---------------------------------------------------------------- config


def test_config_validation_and_overrides(setup, tmp_path):
    cfg = make_cfg(setup, tmp_path, **{"optim.lr": 1e-2, "seq_len": 16})
    assert cfg.optim.lr == 1e-2 and cfg.optim.warmup_steps == 2  # nested override keeps sibling keys
    with pytest.raises(ConfigError, match="unknown key"):
        make_cfg(setup, tmp_path, **{"optim.learning_rate": 1e-3})
    with pytest.raises(ConfigError, match="precision"):
        make_cfg(setup, tmp_path, precision="fp8")
    with pytest.raises(ConfigError, match="min_lr"):
        OptimConfig(lr=1e-4, min_lr=1e-3)
    with pytest.raises(ConfigError, match="run_name"):
        make_cfg(setup, tmp_path, run_name="a/b")
    assert make_cfg(setup, tmp_path, cuda_memory_limit_gib=2.9).cuda_memory_limit_gib == 2.9
    with pytest.raises(ConfigError, match="cuda_memory_limit_gib"):
        make_cfg(setup, tmp_path, cuda_memory_limit_gib=0)


def test_importing_aupus_enables_expandable_segments():
    """Set before PyTorch allocates CUDA memory, so reserved memory can't balloon on Windows (§19)."""
    import os

    import aupus  # noqa: F401  (already imported; the setting happens at first import)

    assert "expandable_segments:True" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")


def test_seq_len_sets_model_context(setup, tmp_path):
    cfg, model_cfg = resolve_model_config(make_cfg(setup, tmp_path, seq_len=24))
    assert model_cfg.context_length == 24 and cfg.tokens_per_step == 4 * 2 * 24
    cfg, model_cfg = resolve_model_config(make_cfg(setup, tmp_path, seq_len=None, model="tiny"))
    assert cfg.seq_len == model_cfg.context_length == ModelConfig.preset("tiny").context_length


def test_shipped_train_configs_load():
    for path in (ROOT / "configs" / "train").glob("*.yaml"):
        cfg, model_cfg = resolve_model_config(TrainConfig.from_yaml(path))
        assert cfg.seq_len == model_cfg.context_length
        assert cfg.optim.warmup_steps < cfg.max_steps


def test_precision_resolution():
    cpu = torch.device("cpu")
    assert resolve_precision("auto", cpu) == "fp32"
    with pytest.raises(RuntimeError, match="CPU"):
        resolve_precision("bf16", cpu)


# ---------------------------------------------------------------- optimizer + schedule


def test_lr_schedule():
    cfg = OptimConfig(lr=1.0, min_lr=0.1, warmup_steps=10)
    lrs = [lr_at(s, cfg, 100) for s in range(100)]
    assert lrs[0] == pytest.approx(0.1) and lrs[9] == pytest.approx(1.0)
    assert all(a < b for a, b in zip(lrs[:10], lrs[1:10], strict=False))  # warmup rises
    assert all(a >= b for a, b in zip(lrs[10:], lrs[11:], strict=False))  # then only falls
    assert lr_at(55, cfg, 100) == pytest.approx(0.55)  # cosine midpoint = (lr + min_lr) / 2
    assert lr_at(100, cfg, 100) == pytest.approx(0.1) and lr_at(500, cfg, 100) == pytest.approx(0.1)


@pytest.mark.parametrize("tie", [True, False])
def test_param_groups_decay_only_linear_weights(tie):
    cfg = ModelConfig(
        vocab_size=50, d_model=16, n_layers=2, n_heads=2, n_kv_heads=1, ffn_hidden=32, tie_embeddings=tie
    )
    model = Transformer(cfg)
    decay, no_decay = param_groups(model, 0.1)
    names = {id(p): n for n, p in model.named_parameters()}
    assert len(decay["params"]) + len(no_decay["params"]) == len(list(model.parameters()))
    assert all(p.ndim == 2 for p in decay["params"])
    assert {names[id(p)] for p in no_decay["params"]} == {
        n for n in names.values() if "norm" in n or n == "embed.weight"
    }
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0


# ---------------------------------------------------------------- training loop


def params_of(trainer: Trainer) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in trainer.model.state_dict().items()}


def test_loss_decreases(setup, tmp_path):
    cfg = make_cfg(setup, tmp_path, max_steps=60, eval_every=60, save_every=0, log_every=10)
    trainer = Trainer(cfg, log=quiet)
    initial_val = trainer.evaluate()
    assert initial_val == pytest.approx(math.log(363), rel=0.05)  # uniform guess at init
    summary = trainer.train()
    assert summary["finished"] and summary["step"] == 60
    assert summary["best_val_loss"] < initial_val - 1.5
    records = [json.loads(line) for line in (cfg.run_dir / "metrics.jsonl").read_text().splitlines()]
    train_losses = [r["train/loss"] for r in records if "train/loss" in r]
    assert train_losses[-1] < train_losses[0] - 1.5
    assert (cfg.run_dir / "step_000060.pt").exists()  # the last step is always saved
    assert (cfg.run_dir / "best.pt").exists() and (cfg.run_dir / "config.yaml").exists()


def test_grad_accumulation_matches_big_batch(setup, tmp_path):
    a = Trainer(make_cfg(setup, tmp_path, run_name="a", micro_batch=4, grad_accum=1, max_steps=1), log=quiet)
    b = Trainer(make_cfg(setup, tmp_path, run_name="b", micro_batch=2, grad_accum=2, max_steps=1), log=quiet)
    a.train()
    b.train()
    pa, pb = params_of(a), params_of(b)
    assert all(torch.allclose(pa[k], pb[k], atol=1e-6) for k in pa)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_resume_on_cuda(setup, tmp_path):
    # Loading onto the GPU also moves the saved RNG states there; resume must still work.
    cfg = make_cfg(setup, tmp_path, device="cuda", max_steps=2, eval_every=0, save_every=0)
    Trainer(cfg, log=quiet).train()
    resumed = Trainer(
        make_cfg(setup, tmp_path, device="cuda", max_steps=3, eval_every=0), resume=True, log=quiet
    )
    assert resumed.step == 2 and resumed.train()["step"] == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_cuda_memory_limit_is_applied(setup, tmp_path):
    cfg = make_cfg(setup, tmp_path, device="cuda", max_steps=1, eval_every=0, cuda_memory_limit_gib=1.0)
    try:
        lines = []
        Trainer(cfg, log=lines.append).train()
        assert any("GPU memory limit 1 GiB" in line for line in lines)
        with pytest.raises(torch.OutOfMemoryError):  # 2 GiB > the 1 GiB limit: refused, not spilled
            torch.empty(2 * 1024**3, dtype=torch.uint8, device="cuda")
    finally:
        torch.cuda.set_per_process_memory_fraction(1.0)
        torch.cuda.empty_cache()


def test_resume_after_ctrl_c_is_bit_identical(setup, tmp_path, monkeypatch):
    full = Trainer(make_cfg(setup, tmp_path, run_name="full"), log=quiet)
    full.train()

    # Simulate Ctrl-C during step 3: the trainer finishes that step, saves and stops.
    interrupted = Trainer(make_cfg(setup, tmp_path, run_name="cut"), log=quiet)
    real_lr_at = pretrain.lr_at

    def lr_at_with_interrupt(step, cfg, max_steps):
        if step == 2:
            interrupted._stop_requested = True
        return real_lr_at(step, cfg, max_steps)

    monkeypatch.setattr(pretrain, "lr_at", lr_at_with_interrupt)
    summary = interrupted.train()
    monkeypatch.setattr(pretrain, "lr_at", real_lr_at)
    assert not summary["finished"] and summary["step"] == 3
    assert ckpt.latest_checkpoint(tmp_path / "cut").name == "step_000003.pt"

    resumed = Trainer(make_cfg(setup, tmp_path, run_name="cut"), resume=True, log=quiet)
    assert resumed.step == 3
    assert resumed.train()["finished"]
    pf, pr = params_of(full), params_of(resumed)
    assert all(torch.equal(pf[k], pr[k]) for k in pf)
    assert (
        resumed.optimizer.state_dict()["state"][0]["step"] == full.optimizer.state_dict()["state"][0]["step"]
    )

    # metrics.jsonl has each step once, as if the run was never interrupted.
    steps = [
        json.loads(line)["step"] for line in (tmp_path / "cut" / "metrics.jsonl").read_text().splitlines()
    ]
    train_steps = [s for s in steps if s > 0]
    assert sorted(set(train_steps)) == list(range(1, 7))


def test_checkpoint_pruning_and_contents(setup, tmp_path):
    cfg = make_cfg(setup, tmp_path, max_steps=8, save_every=2, keep_last=2, eval_every=0)
    Trainer(cfg, log=quiet).train()
    assert [s for s, _ in ckpt.list_checkpoints(cfg.run_dir)] == [6, 8]
    state = ckpt.load(cfg.run_dir / "step_000008.pt")
    assert state["loader"]["num_windows"] > 64  # so 8 steps x 4 x 2 windows stay within epoch 0
    assert state["step"] == 8 and state["loader"] == {**state["loader"], "epoch": 0, "position": 64}
    assert state["train_config"]["run_name"] == "run" and state["model_config"]["context_length"] == 32
    assert not list(cfg.run_dir.glob("*.tmp"))


def test_refuses_to_overwrite_and_checks_resume(setup, tmp_path):
    cfg = make_cfg(setup, tmp_path, max_steps=2, eval_every=0, save_every=0)
    with pytest.raises(FileNotFoundError, match="no checkpoint"):
        Trainer(cfg, resume=True, log=quiet)
    Trainer(cfg, log=quiet).train()
    with pytest.raises(FileExistsError, match="--resume"):
        Trainer(cfg, log=quiet)
    with pytest.raises(ValueError, match="loader state mismatch"):
        Trainer(make_cfg(setup, tmp_path, max_steps=4, micro_batch=2), resume=True, log=quiet)


def test_samples_continue_the_prompt_and_are_reproducible(setup, tmp_path):
    cfg = make_cfg(setup, tmp_path, sample_prompts=["The cat", "A river"], sample_tokens=8)
    trainer = Trainer(cfg, log=quiet)
    first = trainer.samples()
    assert [s[:7] for s in first] == ["The cat", "A river"]
    assert trainer.samples() == first  # fixed seed per prompt


def test_compile_falls_back_to_eager_when_unavailable(setup, tmp_path, monkeypatch):
    monkeypatch.setattr(pretrain, "compile_unsupported_reason", lambda device: "no compiler (test)")
    lines = []
    trainer = Trainer(make_cfg(setup, tmp_path, compile=True), log=lines.append)
    assert trainer.run is trainer.model and not trainer.compiled
    assert any("no compiler (test)" in line for line in lines)


def test_train_script_runs_and_resumes(setup, tmp_path, monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location("train_script", ROOT / "scripts" / "train.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    cfg_path = tmp_path / "run.yaml"
    cfg = make_cfg(setup, tmp_path, max_steps=2, eval_every=0)
    from aupus.config import save_config

    save_config(cfg, cfg_path)
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(cfg_path), "log_every=1"])
    script.main()
    monkeypatch.setattr(sys, "argv", ["train.py", "--resume", str(cfg.run_dir), "max_steps=3"])
    script.main()
    out = capsys.readouterr().out
    assert "resumed from" in out and '"step": 3' in out
