"""Shared fixtures: a tiny trained model, so evaluation, SFT and chat can be tested on CPU in seconds."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from aupus.data import DataConfig, prepare
from aupus.tokenizer import Tokenizer
from aupus.train import TrainConfig, Trainer

WORDS = ["cat", "dog", "sun", "tree", "river", "stone", "light", "house", "water", "garden", "bird", "music"]
VOCAB = 256 + 100 + 7  # bytes + merges + special tokens


def prose(rng: random.Random, n: int = 15) -> str:
    sentences = []
    for _ in range(n):
        body = " ".join(rng.choice(WORDS) for _ in range(rng.randint(6, 12)))
        sentences.append(f"The {body} and that is the way to learn with care.")
    return " ".join(sentences)


@dataclass(frozen=True)
class TinyRun:
    root: Path
    tokenizer: Path
    data_dir: Path
    model_yaml: Path
    run_dir: Path

    @property
    def checkpoint(self) -> Path:
        return self.run_dir / "best.pt"


@pytest.fixture(scope="session")
def tiny_run(tmp_path_factory) -> TinyRun:
    """A 2-layer, 32-wide model trained for 40 steps on generated prose (~2 s on CPU)."""
    root = tmp_path_factory.mktemp("tiny_run")
    rng = random.Random(7)
    docs = [prose(rng) for _ in range(300)]
    tok_path = root / "tok.json"
    Tokenizer.train(docs[:50], vocab_size=VOCAB).save(tok_path)
    prepare(docs, root / "data", tok_path, DataConfig(val_fraction=0.1))
    model_yaml = root / "model.yaml"
    model_yaml.write_text(
        yaml.safe_dump(
            {
                "vocab_size": VOCAB,
                "d_model": 32,
                "n_layers": 2,
                "n_heads": 2,
                "n_kv_heads": 1,
                "ffn_hidden": 64,
            }
        )
    )
    cfg = TrainConfig.from_dict(
        {
            "run_name": "base",
            "out_dir": str(root / "runs"),
            "model": str(model_yaml),
            "data_dir": str(root / "data"),
            "tokenizer": str(tok_path),
            "seq_len": 64,
            "micro_batch": 4,
            "max_steps": 40,
            "optim": {"lr": 3e-3, "min_lr": 3e-4, "warmup_steps": 4},
            "device": "cpu",
            "compile": False,
            "eval_every": 20,
            "eval_batches": 2,
            "save_every": 0,
            "log_every": 10,
        }
    )
    Trainer(cfg, log=lambda _msg: None).train()
    return TinyRun(root, tok_path, root / "data", model_yaml, cfg.run_dir)
