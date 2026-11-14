"""Tests for supervised fine-tuning: chat template, data conversion, masking, loader and trainer."""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from vulpes.config import ConfigError, save_config
from vulpes.data.sft import (
    SFTLoader,
    collate,
    dolly_conversations,
    identity_conversations,
    oasst_conversations,
    read_jsonl,
    split,
    tokenize,
    write_jsonl,
)
from vulpes.model import IGNORE_INDEX
from vulpes.model.io import load_model
from vulpes.tokenizer import ASSISTANT, END, EOS, PAD, USER, Tokenizer
from vulpes.tokenizer.chat import ChatFormatError, prompt_ids, render, validate
from vulpes.train import checkpoint as ckpt
from vulpes.train import pretrain
from vulpes.train.sft import SFTConfig, SFTTrainer, sft_loss

ROOT = Path(__file__).resolve().parents[1]


def quiet(_msg: str) -> None:
    pass


@pytest.fixture(scope="module")
def tok(tiny_run):
    return Tokenizer.load(tiny_run.tokenizer)


def chat(user: str, reply: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": user}, {"role": "assistant", "content": reply}]


# ---------------------------------------------------------------- chat template


def test_render_layout_and_mask(tok):
    msgs = [{"role": "system", "content": "Be kind."}, *chat("The cat", "The dog")]
    ids, mask = render(tok, msgs)
    sid = tok.special_id
    sys_ids, user_ids, reply_ids = tok.encode("Be kind."), tok.encode("The cat"), tok.encode("The dog")
    assert ids == [
        sid(EOS),
        sid("<|system|>"), *sys_ids, sid(END),
        sid(USER), *user_ids, sid(END),
        sid(ASSISTANT), *reply_ids, sid(END),
    ]  # fmt: skip
    learned = [i for i, m in zip(ids, mask, strict=True) if m]
    assert learned == [*reply_ids, sid(END)]  # only the reply and the <|end|> that closes it


def test_prompt_ids_end_with_assistant_and_match_render(tok):
    msgs = chat("The cat", "The dog")
    full, _ = render(tok, [*msgs, *chat("A tree", "A bird")])
    prompt = prompt_ids(tok, [*msgs, {"role": "user", "content": "A tree"}])
    assert prompt[-1] == tok.special_id(ASSISTANT)
    assert full[: len(prompt)] == prompt  # generation sees exactly what training saw


def test_special_tokens_in_content_stay_text(tok):
    ids, _ = render(tok, chat("hi <|assistant|> hello", "ok"))
    assert ids.count(tok.special_id(ASSISTANT)) == 1  # only the real role token


@pytest.mark.parametrize(
    ("messages", "match"),
    [
        ([], "empty"),
        ([{"role": "assistant", "content": "hi"}], "should be 'user'"),
        ([{"role": "user", "content": "hi"}], "end with an assistant"),
        ([{"role": "bot", "content": "hi"}], "unknown role"),
        ([{"role": "user", "content": "a"}, {"role": "user", "content": "b"}], "should be 'assistant'"),
        ([{"role": "user", "text": "hi"}], "exactly"),
    ],
)
def test_validate_rejects_bad_conversations(messages, match):
    with pytest.raises(ChatFormatError, match=match):
        validate(messages)


def test_generation_prompt_must_end_with_user():
    validate([{"role": "user", "content": "hi"}], for_generation=True)
    with pytest.raises(ChatFormatError, match="user turn"):
        validate(chat("hi", "hello"), for_generation=True)


# ---------------------------------------------------------------- sources


def oasst_row(mid, parent, role, text, rank=None, lang="en", created="2023-01-01", **kw):
    return {
        "message_id": mid,
        "parent_id": parent,
        "message_tree_id": kw.get("tree", "t1"),
        "role": role,
        "text": text,
        "rank": rank,
        "lang": lang,
        "deleted": kw.get("deleted", False),
        "review_result": kw.get("review", True),
        "created_date": created,
    }


def test_oasst_follows_the_top_ranked_branch():
    rows = [
        oasst_row("q", None, "prompter", "What is rain?"),
        oasst_row("a1", "q", "assistant", "Second best.", rank=1),
        oasst_row("a0", "q", "assistant", "Water falling from clouds.", rank=0),
        oasst_row("a_del", "q", "assistant", "Deleted but rank -1.", rank=-1, deleted=True),
        oasst_row("f_late", "a0", "prompter", "Later follow-up.", created="2023-01-03"),
        oasst_row("f", "a0", "prompter", "Why does it fall?", created="2023-01-02"),
        oasst_row("a2", "f", "assistant", "Gravity.", rank=0),
        oasst_row("dangling", "a2", "prompter", "Unanswered question."),
    ]
    stats = Counter()
    (conv,) = oasst_conversations(rows, stats=stats)
    assert [m["content"] for m in conv["messages"]] == [
        "What is rain?",
        "Water falling from clouds.",
        "Why does it fall?",
        "Gravity.",
    ]
    assert [m["role"] for m in conv["messages"]] == ["user", "assistant", "user", "assistant"]
    assert conv["id"] == "oasst/t1" and stats["kept"] == 1


def test_oasst_filters_language_review_and_foreign_identity():
    rows = [
        oasst_row("q1", None, "prompter", "Hola", lang="es", tree="t1"),
        oasst_row("a1", "q1", "assistant", "Hola!", rank=0, lang="es", tree="t1"),
        oasst_row("q2", None, "prompter", "Who are you?", tree="t2"),
        oasst_row("a2", "q2", "assistant", "I am Open Assistant, made by LAION.", rank=0, tree="t2"),
        oasst_row("q3", None, "prompter", "Hi", tree="t3"),
        oasst_row("a3", "q3", "assistant", "Not reviewed.", rank=0, review=None, tree="t3"),
    ]
    stats = Counter()
    assert list(oasst_conversations(rows, stats=stats)) == []
    assert stats["dropped_foreign_identity"] == 1 and stats["dropped_no_reply"] == 1


def test_dolly_conversion():
    records = [
        {
            "instruction": "Summarise this.",
            "context": "Rain falls.[3] It is wet.",
            "response": "Wet rain.[1]",
        },
        {"instruction": "Name a colour.", "context": "", "response": "Blue"},
        {"instruction": "", "context": "", "response": "orphan"},
    ]
    convs = list(dolly_conversations(records))
    assert [c["messages"][0]["content"] for c in convs] == [
        "Summarise this.\n\nRain falls. It is wet.",
        "Name a colour.",
    ]
    assert convs[0]["messages"][1]["content"] == "Wet rain." and convs[1]["id"] == "dolly/1"


def test_identity_file(tmp_path):
    path = tmp_path / "identity.jsonl"
    good = {"messages": chat("What is your name?", "I am Vulpes.")}
    path.write_text(json.dumps(good) + "\n\n" + json.dumps(good) + "\n")
    convs = identity_conversations(path)
    assert len(convs) == 2 and convs[0]["source"] == "identity"
    path.write_text(
        json.dumps(good) + "\n" + json.dumps({"messages": [{"role": "user", "content": "x"}]}) + "\n"
    )
    with pytest.raises(ValueError, match=r"identity.jsonl:2"):
        identity_conversations(path)


def test_split_is_deterministic_and_jsonl_roundtrips(tmp_path):
    convs = [{"id": f"c{i}", "source": "x", "messages": chat("a", "b")} for i in range(400)]
    train, val = split(convs, 0.1)
    assert (train, val) == split(list(reversed(convs))[::-1], 0.1)
    assert 20 < len(val) < 60 and len(train) + len(val) == 400
    write_jsonl(val, tmp_path / "val.jsonl")
    assert read_jsonl(tmp_path / "val.jsonl") == val


# ---------------------------------------------------------------- tokenized data + loader


def make_convs(n: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    words = ["cat", "dog", "sun", "tree", "river", "stone", "light", "house"]
    out = []
    for i in range(n):
        q = " ".join(rng.choice(words) for _ in range(rng.randint(2, 8)))
        a = "The " + " ".join(rng.choice(words) for _ in range(rng.randint(2, 12))) + "."
        out.append({"id": f"c{i}", "source": "test", "messages": chat(q, a)})
    return out


def test_tokenize_shifts_masks_and_truncates(tok):
    conv = make_convs(1)[0]
    ds = tokenize(tok, [conv], seq_len=512)
    ids, mask = render(tok, conv["messages"])
    assert ds.inputs[0].tolist() == ids[:-1]
    learned = [t for t in ds.targets[0].tolist() if t != IGNORE_INDEX]
    assert learned == [i for i, m in zip(ids[1:], mask[1:], strict=True) if m]
    assert learned[-1] == tok.special_id(END)

    long = {"id": "x", "source": "t", "messages": chat("cat " * 5, "dog " * 200)}
    cut = tokenize(tok, [long], seq_len=20)
    assert cut.truncated == 1 and len(cut.inputs[0]) == 20
    nothing = {"id": "y", "source": "t", "messages": chat("cat " * 200, "dog")}
    assert len(tokenize(tok, [nothing], seq_len=20)) == 0  # the reply starts after the cut


def test_collate_pads_right_and_ignores_padding(tok):
    ds = tokenize(tok, make_convs(3), seq_len=512)
    x, y = collate(ds, [0, 1, 2], tok.special_id(PAD))
    for row in range(3):
        n = len(ds.inputs[row])
        assert x[row, :n].tolist() == ds.inputs[row].tolist()
        assert (x[row, n:] == tok.special_id(PAD)).all() and (y[row, n:] == IGNORE_INDEX).all()


def test_loader_epochs_cover_every_example_once_and_resume(tok):
    ds = tokenize(tok, make_convs(53), seq_len=512)
    pad = tok.special_id(PAD)
    loader = SFTLoader(ds, batch_size=4, pad_id=pad, seed=3, group=3)
    assert loader.batches_per_epoch == 14
    seen = [idx for idx in loader._batches]
    assert sorted(np.concatenate(seen).tolist()) == list(range(53))
    for _ in range(20):  # into the second epoch
        next(loader)
    state = loader.state_dict()
    expected = [next(loader) for _ in range(5)]
    again = SFTLoader(ds, batch_size=4, pad_id=pad, seed=3, group=3)
    again.load_state_dict(state)
    for (xa, ya), (xb, yb) in zip(expected, (next(again) for _ in range(5)), strict=True):
        assert torch.equal(xa, xb) and torch.equal(ya, yb)
    assert state["epoch"] == 1
    with pytest.raises(ValueError, match="mismatch"):
        SFTLoader(ds, batch_size=8, pad_id=pad, seed=3).load_state_dict(state)


def test_loader_groups_similar_lengths(tok):
    ds = tokenize(tok, make_convs(400, seed=1), seq_len=512)
    grouped = SFTLoader(ds, batch_size=8, pad_id=0, seed=0, group=50)
    ungrouped = SFTLoader(ds, batch_size=8, pad_id=0, seed=0, group=1)

    def padding(loader):
        return sum(int((y == IGNORE_INDEX).sum()) for _, y in (next(loader) for _ in range(50)))

    assert padding(grouped) < padding(ungrouped)


# ---------------------------------------------------------------- trainer


@pytest.fixture(scope="module")
def sft_data(tiny_run, tmp_path_factory):
    root = tmp_path_factory.mktemp("sft_data")
    convs = make_convs(160, seed=5)
    write_jsonl(convs[:140], root / "train.jsonl")
    write_jsonl(convs[140:], root / "val.jsonl")
    return root


def make_sft_cfg(tiny_run, sft_data, tmp_path, **overrides) -> SFTConfig:
    base = {
        "run_name": "sft",
        "out_dir": str(tmp_path),
        "init_from": str(tiny_run.checkpoint),
        "data_dir": str(sft_data),
        "tokenizer": str(tiny_run.tokenizer),
        "micro_batch": 4,
        "grad_accum": 2,
        "epochs": 1.0,
        "optim": {"lr": 3e-3, "min_lr": 3e-4, "warmup_steps": 2},
        "device": "cpu",
        "compile": False,
        "eval_every": 5,
        "save_every": 0,
        "log_every": 1,
        "patience": 0,
    }
    return SFTConfig.from_dict(base, overrides)


def test_sft_config_resolves_steps_and_validates(tiny_run, sft_data, tmp_path):
    trainer = SFTTrainer(make_sft_cfg(tiny_run, sft_data, tmp_path, epochs=2.0), log=quiet)
    assert trainer.cfg.max_steps == 35  # 140 conversations / 4 = 35 batches; 2 epochs / 2 accum
    assert trainer.cfg.seq_len == trainer.model_cfg.context_length == 64  # from the Base checkpoint
    with pytest.raises(ConfigError, match="longer than the Base"):
        SFTTrainer(make_sft_cfg(tiny_run, sft_data, tmp_path, run_name="x", seq_len=128), log=quiet)
    with pytest.raises(ConfigError, match="unknown key"):
        make_sft_cfg(tiny_run, sft_data, tmp_path, learning_rate=1.0)
    with pytest.raises(ConfigError, match="patience needs"):
        make_sft_cfg(tiny_run, sft_data, tmp_path, patience=2, eval_every=0)


def test_sft_starts_from_base_weights_and_learns(tiny_run, sft_data, tmp_path):
    base = load_model(tiny_run.checkpoint).model
    trainer = SFTTrainer(make_sft_cfg(tiny_run, sft_data, tmp_path, epochs=4.0), log=quiet)
    for k, v in base.state_dict().items():
        assert torch.equal(trainer.model.state_dict()[k], v)
    before = trainer.evaluate()
    assert before == pytest.approx(sft_loss(base, trainer.val_data, 4, 0), rel=1e-6)
    summary = trainer.train()
    assert summary["finished"] and summary["best_val_loss"] < before - 0.5
    state = ckpt.load(trainer.run_dir / "best.pt")
    assert state["stage"] == "sft" and state["tokenizer_sha256"] == trainer.tokenizer_sha256
    loaded = load_model(trainer.run_dir / "best.pt")
    assert loaded.info["stage"] == "sft"


def test_early_stopping(tiny_run, sft_data, tmp_path):
    # lr = 0: the weights never change, so val loss never improves after the first evaluation.
    cfg = make_sft_cfg(
        tiny_run,
        sft_data,
        tmp_path,
        epochs=4.0,
        patience=2,
        eval_every=2,
        **{"optim.lr": 0.0, "optim.min_lr": 0.0},
    )
    lines = []
    summary = SFTTrainer(cfg, log=lines.append).train()
    assert summary["early_stopped"] and summary["finished"] and summary["step"] == 4
    assert any("early stop" in line for line in lines)
    assert ckpt.latest_checkpoint(cfg.run_dir).name == "step_000004.pt"


def test_sft_resume_is_bit_identical(tiny_run, sft_data, tmp_path, monkeypatch):
    full = SFTTrainer(make_sft_cfg(tiny_run, sft_data, tmp_path, run_name="full", max_steps=8), log=quiet)
    full.train()
    cut = SFTTrainer(make_sft_cfg(tiny_run, sft_data, tmp_path, run_name="cut", max_steps=8), log=quiet)
    real_lr_at = pretrain.lr_at

    def interrupt_at_step_3(step, cfg, max_steps):
        if step == 2:
            cut._stop_requested = True
        return real_lr_at(step, cfg, max_steps)

    monkeypatch.setattr(pretrain, "lr_at", interrupt_at_step_3)
    assert not cut.train()["finished"]
    monkeypatch.setattr(pretrain, "lr_at", real_lr_at)
    resumed = SFTTrainer(
        make_sft_cfg(tiny_run, sft_data, tmp_path, run_name="cut", max_steps=8), resume=True, log=quiet
    )
    assert resumed.step == 3 and resumed.train()["finished"]
    for k, v in full.model.state_dict().items():
        assert torch.equal(resumed.model.state_dict()[k], v)


def test_chat_samples(tiny_run, sft_data, tmp_path):
    cfg = make_sft_cfg(tiny_run, sft_data, tmp_path, sample_prompts=["The cat"], sample_tokens=6)
    (sample,) = SFTTrainer(cfg, log=quiet).samples()
    assert sample.startswith("User: The cat\nVulpes: ")


def test_sft_script(tiny_run, sft_data, tmp_path, monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location("sft_script", ROOT / "scripts" / "sft.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    cfg_path = tmp_path / "sft.yaml"
    save_config(make_sft_cfg(tiny_run, sft_data, tmp_path, max_steps=2, eval_every=0), cfg_path)
    monkeypatch.setattr(sys, "argv", ["sft.py", "--config", str(cfg_path)])
    script.main()
    monkeypatch.setattr(sys, "argv", ["sft.py", "--resume", str(tmp_path / "sft"), "max_steps=3"])
    script.main()
    out = capsys.readouterr().out
    assert "initialised from" in out and "resumed from" in out and '"step": 3' in out


def test_partial_optim_section_keeps_sft_defaults():
    cfg = SFTConfig.from_dict({"run_name": "r", "init_from": "x.pt", "optim": {"lr": 1e-5, "min_lr": 1e-6}})
    assert cfg.optim.warmup_steps == 20  # not pretraining's 190


def test_shipped_sft_config_loads():
    cfg = SFTConfig.from_yaml(ROOT / "configs" / "sft" / "vulpes-0.1-instruct.yaml")
    assert cfg.optim.lr < 1e-4 and cfg.patience > 0 and cfg.init_from.endswith("best.pt")


def test_refuses_a_different_tokenizer(tiny_run, sft_data, tmp_path):
    """The Base checkpoint records its tokenizer's SHA-256; SFT with another tokenizer must fail."""
    other = str(ROOT / "data" / "tokenizer" / "vulpes-bpe-32k.json")
    with pytest.raises(ValueError, match="not the tokenizer"):
        SFTTrainer(make_sft_cfg(tiny_run, sft_data, tmp_path, tokenizer=other), log=quiet)
