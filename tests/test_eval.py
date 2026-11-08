"""Tests for evaluation: task formats, continuation scoring, accuracy, perplexity and the CLI."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from aupus.data import eval_batches
from aupus.eval import (
    Example,
    encode_pair,
    evaluate_loss,
    read_examples,
    run_task,
    score_continuations,
    write_examples,
)
from aupus.eval import scoring as scoring_module
from aupus.eval.tasks import arc, hellaswag, piqa
from aupus.model import ModelConfig, Transformer
from aupus.model.io import load_model
from aupus.tokenizer import EOS, Tokenizer

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- task formats


def test_hellaswag_format():
    record = {
        "ind": 7,
        "activity_label": "Making tea",
        "ctx_a": "A woman fills a kettle.",
        "ctx_b": "she",
        # Some exports store the list as its Python repr.
        "endings": "['puts it on the [title] stove.', 'drinks the  kettle.']",
        "label": "0",
    }
    ex = hellaswag(record)
    assert ex.context == "Making tea: A woman fills a kettle. She"
    assert ex.choices == [" puts it on the. stove.", " drinks the kettle."]
    assert ex.label == 0
    assert hellaswag({**record, "endings": ["a", "b"]}).choices == [" a", " b"]


def test_arc_format_with_letter_and_number_keys():
    record = {
        "id": "q1",
        "question": "What do plants need?",
        "choices": {"text": ["Light", "Rocks", "Noise"], "label": ["A", "B", "C"]},
        "answerKey": "A",
    }
    ex = arc(record)
    assert ex.context == "Question: What do plants need?\nAnswer:" and ex.choices[0] == " Light"
    assert ex.label == 0
    numbered = {**record, "choices": {"text": ["x", "y"], "label": ["1", "2"]}, "answerKey": "2"}
    assert arc(numbered).label == 1


def test_piqa_format_and_roundtrip(tmp_path):
    ex = piqa({"goal": "Open a jar", "sol1": "Twist the lid.", "sol2": "Eat the lid.", "label": "0"})
    assert ex.context == "Question: Open a jar\nAnswer:" and ex.choices == [
        " Twist the lid.",
        " Eat the lid.",
    ]
    write_examples([ex, ex], tmp_path / "piqa.jsonl")
    assert list(read_examples(tmp_path / "piqa.jsonl")) == [ex, ex]


def test_example_rejects_bad_label():
    with pytest.raises(ValueError, match="label"):
        Example("x", "ctx", [" a", " b"], 2)


# ---------------------------------------------------------------- scoring


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=64, d_model=32, n_layers=2, n_heads=2, n_kv_heads=1, ffn_hidden=64, context_length=16
    )
    return Transformer(cfg).eval()


@torch.no_grad()
def manual_logprob(model, ctx, cont):
    seq = torch.tensor([ctx + cont])
    logp = F.log_softmax(model(seq[:, :-1]).logits[0].float(), dim=-1)
    return sum(float(logp[len(ctx) - 1 + i, t]) for i, t in enumerate(cont))


def test_scores_match_a_manual_forward_pass(model):
    requests = [([1, 2, 3], [4, 5]), ([7], [8, 9, 10, 11]), ([3, 3, 3, 3, 3], [2])]
    scores = score_continuations(model, requests, batch_size=1)
    for (ctx, cont), s in zip(requests, scores, strict=True):
        assert s.logprob == pytest.approx(manual_logprob(model, ctx, cont), abs=1e-5)
        assert s.logprob < 0


def test_batching_and_padding_do_not_change_scores(model):
    g = torch.Generator().manual_seed(1)
    requests = []
    for _ in range(9):
        n_ctx, n_cont = (int(v) for v in torch.randint(1, 7, (2,), generator=g))
        requests.append(
            (
                torch.randint(0, 64, (n_ctx,), generator=g).tolist(),
                torch.randint(0, 64, (n_cont,), generator=g).tolist(),
            )
        )
    one = score_continuations(model, requests, batch_size=1)
    many = score_continuations(model, requests, batch_size=4)
    for a, b in zip(one, many, strict=True):
        assert a.logprob == pytest.approx(b.logprob, abs=1e-5) and a.greedy == b.greedy


def test_greedy_flag(model):
    ctx = [5, 6, 7]
    with torch.no_grad():
        best = int(model(torch.tensor([ctx])).logits[0, -1].argmax())
    other = (best + 1) % 64
    greedy, not_greedy = score_continuations(model, [(ctx, [best]), (ctx, [other])])
    assert greedy.greedy and not not_greedy.greedy and greedy.logprob > not_greedy.logprob


def test_long_requests_drop_the_start_of_the_context(model):
    ctx, cont = list(range(30)), [40, 41]  # 32 tokens > window of 16 + 1
    (s,) = score_continuations(model, [(ctx, cont)])
    kept = (ctx + cont)[-17:]
    assert s.logprob == pytest.approx(manual_logprob(model, kept[:-2], cont), abs=1e-5)
    with pytest.raises(ValueError, match="continuation"):
        score_continuations(model, [([1], [])])


def test_encode_pair_splits_cleanly(tiny_run):
    tok = Tokenizer.load(tiny_run.tokenizer)
    ctx, cont = encode_pair(tok, "Question: The cat and the dog\nAnswer:", " The river")
    assert ctx[0] == tok.special_id(EOS)
    assert ctx[1:] + cont == tok.encode("Question: The cat and the dog\nAnswer: The river")
    assert tok.decode(cont) == " The river"


def test_accuracy_and_length_normalisation(monkeypatch, tiny_run):
    """acc picks the highest log-prob; acc_norm the highest log-prob per character."""
    tok = Tokenizer.load(tiny_run.tokenizer)
    examples = [
        Example("a", "ctx", [" x", " a much longer answer"], 1),  # long answer: raw loses, normalised wins
        Example("b", "ctx", [" y", " z"], 0),
    ]
    fake = iter([-2.0, -10.0, -1.0, -3.0])

    def fake_scores(model, requests, batch_size=16, progress=False):
        return [scoring_module.Score(next(fake), False) for _ in requests]

    monkeypatch.setattr(scoring_module, "score_continuations", fake_scores)
    r = run_task("test", None, tok, examples)
    assert (r.n, r.acc, r.acc_norm) == (2, 0.5, 1.0)
    assert r.acc_stderr == pytest.approx(math.sqrt(0.25 / 2))


def test_trained_model_prefers_real_text(tiny_run):
    """After 40 steps on the generated prose, its own kind of sentence beats shuffled letters."""
    loaded = load_model(tiny_run.checkpoint)
    examples = [
        Example(
            f"e{i}",
            "The cat and the dog",
            [" and that is the way to learn with care.", " ecar htiw nrael"],
            0,
        )
        for i in range(3)
    ]
    r = run_task("test", loaded.model, loaded.tokenizer, examples)
    assert r.acc == 1.0 and r.acc_norm == 1.0


# ---------------------------------------------------------------- perplexity


def test_evaluate_loss_matches_mean_cross_entropy(tiny_run):
    loaded = load_model(tiny_run.checkpoint)
    model, tok = loaded.model, loaded.tokenizer
    out = evaluate_loss(model, tok, tiny_run.data_dir, batch_size=4, max_batches=3)
    losses, tokens, nbytes = [], 0, 0
    with torch.no_grad():
        for x, y in eval_batches(tiny_run.data_dir, "val", 4, 64, 3):
            losses.append(float(model(x, y).loss) * y.numel())
            tokens += y.numel()
            nbytes += sum(0 if tok.is_special(i) else len(tok.token_bytes(i)) for i in y.flatten().tolist())
    assert out["tokens"] == tokens
    assert out["loss"] == pytest.approx(sum(losses) / tokens, rel=1e-5)
    assert out["ppl"] == pytest.approx(math.exp(out["loss"]), rel=1e-6)
    assert out["bits_per_byte"] == pytest.approx(sum(losses) / math.log(2) / nbytes, rel=1e-5)


def test_untrained_model_perplexity_is_about_vocab_size(tiny_run):
    tok = Tokenizer.load(tiny_run.tokenizer)
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=tok.vocab_size,
        d_model=32,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        ffn_hidden=64,
        context_length=64,
    )
    out = evaluate_loss(Transformer(cfg), tok, tiny_run.data_dir, max_batches=2)
    assert out["ppl"] == pytest.approx(tok.vocab_size, rel=0.1)


# ---------------------------------------------------------------- CLI


def test_evaluate_script(tiny_run, tmp_path):
    ex = Example("e", "The cat", [" and the dog", " zzzz"], 0)
    for task in ("piqa", "arc_easy"):
        write_examples([ex] * 4, tmp_path / "eval" / f"{task}.jsonl")
    out = tmp_path / "results.json"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate.py"),
        "--model", str(tiny_run.checkpoint),
        "--tasks", "piqa,arc_easy",
        "--eval-dir", str(tmp_path / "eval"),
        "--val-data", str(tiny_run.data_dir),
        "--val-batches", "2",
        "--device", "cpu",
        "--out", str(out),
    ]  # fmt: skip
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, check=False)
    assert proc.returncode == 0, proc.stderr
    results = json.loads(out.read_text())
    assert set(results["tasks"]) == {"piqa", "arc_easy"} and results["tasks"]["piqa"]["n"] == 4
    assert results["val"]["loss"] > 0 and results["info"]["step"] == 40
