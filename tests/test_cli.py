"""Tests for inference: release export/load, streaming text, repetition penalty, generate and chat."""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from vulpes.cli import chat as chat_cli
from vulpes.cli import generate as generate_cli
from vulpes.cli.streaming import TextStream
from vulpes.cli.term import ANSI, ReplyWriter, Term, cell_width, clip, wrap
from vulpes.model import ModelConfig, Transformer, generate
from vulpes.model.generate import apply_repetition_penalty
from vulpes.model.io import export_release, load_model
from vulpes.tokenizer import EOS, Tokenizer
from vulpes.tokenizer.chat import prompt_ids

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- release format


def logits_of(model, n=20):
    x = torch.arange(n)[None] % model.cfg.vocab_size
    with torch.no_grad():
        return model(x).logits


def test_export_roundtrip_is_exact_in_fp32(tiny_run, tmp_path):
    out = export_release(tiny_run.checkpoint, tmp_path / "rel")
    assert sorted(p.name for p in out.iterdir()) == [
        "model.safetensors",
        "model.yaml",
        "release.json",
        "tokenizer.json",
    ]
    info = json.loads((out / "release.json").read_text())
    assert info["step"] == 40 and info["stage"] == "pretrain" and info["dtype"] == "fp32"
    assert set(info["sha256"]) == {"model.safetensors", "model.yaml", "tokenizer.json"}
    ckpt, rel = load_model(tiny_run.checkpoint), load_model(out)
    assert torch.equal(logits_of(ckpt.model), logits_of(rel.model))
    assert rel.tokenizer.encode("The cat") == ckpt.tokenizer.encode("The cat")
    assert rel.info["kind"] == "release" and not rel.model.training


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_half_precision_export_is_smaller_and_close(tiny_run, tmp_path, dtype):
    full = export_release(tiny_run.checkpoint, tmp_path / "fp32")
    half = export_release(tiny_run.checkpoint, tmp_path / dtype, dtype=dtype)
    assert (half / "model.safetensors").stat().st_size < 0.6 * (full / "model.safetensors").stat().st_size
    diff = (logits_of(load_model(full).model) - logits_of(load_model(half).model)).abs().max()
    assert 0 < diff < 0.1


def test_release_from_before_the_rename_still_loads(tiny_run, tmp_path):
    """Releases exported while the project was called Aupus say ``aupus-release``."""
    out = export_release(tiny_run.checkpoint, tmp_path / "rel")
    info = json.loads((out / "release.json").read_text())
    assert info["format"] == "vulpes-release"
    (out / "release.json").write_text(json.dumps(info | {"format": "aupus-release"}))
    assert load_model(out).info["kind"] == "release"
    (out / "release.json").write_text(json.dumps(info | {"format": "other"}))
    with pytest.raises(ValueError, match="not a vulpes-release"):
        load_model(out)


def test_export_refuses_to_overwrite(tiny_run, tmp_path):
    export_release(tiny_run.checkpoint, tmp_path / "rel")
    with pytest.raises(FileExistsError, match="not empty"):
        export_release(tiny_run.checkpoint, tmp_path / "rel")
    export_release(tiny_run.checkpoint, tmp_path / "rel", overwrite=True)
    with pytest.raises(ValueError, match="dtype"):
        export_release(tiny_run.checkpoint, tmp_path / "other", dtype="int8")


def test_missing_model_path_gives_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="model not found.*repository folder"):
        load_model(tmp_path / "release" / "nope")


def test_load_checks_the_tokenizer(tiny_run):
    other = ROOT / "data" / "tokenizer" / "vulpes-bpe-32k.json"
    with pytest.raises(ValueError, match="SHA-256"):
        load_model(tiny_run.checkpoint, tokenizer_path=other)


def test_export_script(tiny_run, tmp_path):
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "export_release.py"),
        "--checkpoint",
        str(tiny_run.checkpoint),
    ]
    proc = subprocess.run([*cmd, "--out", str(tmp_path / "rel")], capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 0, proc.stderr
    assert "max logit difference vs checkpoint: 0.00e+00" in proc.stdout
    assert (tmp_path / "rel" / "LICENSE").exists()


# ---------------------------------------------------------------- streaming + sampling


def test_text_stream_never_splits_a_character(tiny_run):
    tok = Tokenizer.load(tiny_run.tokenizer)
    ids = tok.encode("café 🙂 ok")  # é and 🙂 are multi-byte; this small vocab keeps them as byte tokens
    stream = TextStream(tok)
    pieces = [stream.push(i) for i in ids]
    assert "".join(pieces) + stream.flush() == "café 🙂 ok"
    assert "�" not in "".join(pieces)
    assert "" in pieces  # some byte tokens completed nothing on their own
    assert stream.push(tok.special_id(EOS)) == ""


def test_repetition_penalty():
    logits = torch.tensor([2.0, -2.0, 1.0, 0.5])
    out = apply_repetition_penalty(logits, torch.tensor([0, 1]), 2.0)
    assert out.tolist() == [1.0, -4.0, 1.0, 0.5]  # both seen tokens become less likely
    assert torch.equal(apply_repetition_penalty(logits, torch.tensor([0]), 1.0), logits)


def test_strong_penalty_stops_greedy_repetition():
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=50, d_model=32, n_layers=2, n_heads=2, n_kv_heads=1, ffn_hidden=64, context_length=64
    )
    model = Transformer(cfg)
    plain = generate(model, [1, 2, 3], 30, temperature=0)
    assert len(set(plain)) < len(plain)  # an untrained model repeats itself
    penalised = generate(model, [1, 2, 3], 30, temperature=0, repetition_penalty=1e6)
    assert len(set(penalised)) == len(penalised) and not set(penalised) & {1, 2, 3}
    with pytest.raises(ValueError, match="repetition_penalty"):
        generate(model, [1], 3, repetition_penalty=0)


def test_vocab_limit_never_samples_ids_the_tokenizer_lacks(tiny_run):
    """The test model's embedding (363 ids) is bigger than its tokenizer: those ids have no text."""
    loaded = load_model(tiny_run.checkpoint)
    tok = loaded.tokenizer
    assert loaded.model.cfg.vocab_size > tok.vocab_size
    g = torch.Generator().manual_seed(0)
    free = generate(loaded.model, [1, 2], 60, temperature=50.0, generator=g)
    assert max(free) >= tok.vocab_size  # without the limit, a near-uniform sample hits them
    g = torch.Generator().manual_seed(0)
    limited = generate(loaded.model, [1, 2], 60, temperature=50.0, generator=g, vocab_limit=tok.vocab_size)
    assert max(limited) < tok.vocab_size
    tok.decode(limited)  # decodable, which is what crashed the chat before


# ---------------------------------------------------------------- generate


def test_generate_cli_is_reproducible_and_greedy_matches_the_library(tiny_run):
    def run(*extra):
        out = io.StringIO()
        generate_cli.main(
            ["--model", str(tiny_run.checkpoint), "--prompt", "The cat", "--device", "cpu", *extra], out
        )
        return out.getvalue()

    seeded = run("--seed", "3", "-n", "12")
    assert seeded.startswith("The cat") and seeded == run("--seed", "3", "-n", "12")
    greedy = run("--temperature", "0", "--repetition-penalty", "1", "-n", "12")
    loaded = load_model(tiny_run.checkpoint)
    tok = loaded.tokenizer
    eos = tok.special_id(EOS)
    expected = generate(loaded.model, [eos, *tok.encode("The cat")], 12, temperature=0, stop_ids=[eos])
    assert greedy == "The cat" + tok.decode(expected, skip_special=True) + "\n"


# ---------------------------------------------------------------- chat


def chat_args(**kw):
    defaults = {
        "system": None,
        "max_new_tokens": 8,
        "temperature": 0.0,
        "top_k": 0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "seed": None,
    }
    return argparse.Namespace(**{**defaults, **kw})


def scripted(lines):
    it = iter(lines)

    def read_line(_prompt):
        try:
            return next(it)
        except StopIteration:
            raise EOFError from None

    return read_line


def test_chat_loop_keeps_history_and_handles_commands(tiny_run):
    loaded = load_model(tiny_run.checkpoint)
    out = io.StringIO()
    inputs = ["The cat", "", "/nope", "The dog", "/reset", "A tree", "/exit", "never read"]
    history = chat_cli.chat_loop(loaded, chat_args(system="Be brief."), scripted(inputs), out)
    text = out.getvalue()
    assert "Base model" in text and "unknown command /nope" in text and "(conversation cleared)" in text
    assert [m["role"] for m in history] == ["system", "user", "assistant"]
    assert history[0]["content"] == "Be brief." and history[1]["content"] == "A tree"


def test_chat_reply_equals_greedy_generation_from_the_template(tiny_run):
    loaded = load_model(tiny_run.checkpoint)
    tok = loaded.tokenizer
    messages = [{"role": "user", "content": "The cat"}]
    reply, dropped = chat_cli.respond(loaded, messages, 8, io.StringIO(), temperature=0.0)
    ids = generate(loaded.model, prompt_ids(tok, messages), 8, temperature=0)
    assert dropped == 0 and reply == tok.decode(ids, skip_special=True).strip()


def test_fit_history_drops_oldest_exchanges_first(tiny_run):
    tok = Tokenizer.load(tiny_run.tokenizer)
    long = "The cat and the dog " * 5
    messages = [{"role": "system", "content": "Hi."}]
    for i in range(6):
        messages += [{"role": "user", "content": f"{long} {i}"}, {"role": "assistant", "content": long}]
    messages.append({"role": "user", "content": "last"})
    ids, kept = chat_cli.fit_history(tok, messages, window=200, reserve=20)
    assert len(ids) + 20 <= 200
    assert kept[0] == messages[0] and kept[-1] == messages[-1]  # system and newest turn survive
    assert kept[1:] == messages[len(messages) - len(kept) + 1 :]  # what's kept is the most recent run
    with pytest.raises(ValueError, match="too long"):
        chat_cli.fit_history(tok, [{"role": "user", "content": long * 20}], window=64, reserve=8)


def test_chat_reports_a_message_that_cannot_fit(tiny_run):
    loaded = load_model(tiny_run.checkpoint)  # context window 64
    out = io.StringIO()
    history = chat_cli.chat_loop(loaded, chat_args(), scripted(["The cat " * 100, "The dog"]), out)
    assert "too long" in out.getvalue()
    assert [m["content"] for m in history if m["role"] == "user"] == ["The dog"]


def test_chat_retry_and_save(tiny_run, tmp_path):
    loaded = load_model(tiny_run.checkpoint)
    out = io.StringIO()
    inputs = ["/retry", "/save", "The cat", "/retry", "/save"]
    history = chat_cli.chat_loop(loaded, chat_args(save_dir=tmp_path), scripted(inputs), out)
    text = out.getvalue()
    assert "Nothing to retry yet" in text and "Nothing to save yet" in text
    assert [m["role"] for m in history] == ["user", "assistant"]  # /retry replaced the reply
    (saved,) = tmp_path.glob("chat-*.jsonl")
    assert json.loads(saved.read_text(encoding="utf-8"))["messages"] == history


def test_chat_settings_change_the_next_replies(tiny_run):
    loaded = load_model(tiny_run.checkpoint)  # context window 64: a reply gets at most 32 tokens
    out = io.StringIO()
    args = chat_args()
    inputs = ["/settings", "/settings reply length 3", "/settings TEMP 0,5", "/settings temperature 0"]
    inputs += [
        "/settings top-k 1.5",
        "/settings length 33",
        "/settings volume 3",
        "/settings top_p",
        "The cat",
    ]
    history = chat_cli.chat_loop(loaded, args, scripted(inputs), out)
    text = out.getvalue()
    assert "Repetition penalty" in text and "1 to 32" in text  # the table, with each setting's range
    assert "Reply length is now 3 (it was 8)" in text and "Temperature is now 0.5 (it was 0)" in text
    assert "Top-k is a whole number" in text and "Reply length goes from 1 to 32" in text
    assert "no setting called 'volume'" in text and "Top-p is 1 (it goes from 0.05 to 1)" in text
    assert (args.max_new_tokens, args.temperature, args.top_k) == (3, 0.0, 0)
    messages = [{"role": "user", "content": "The cat"}]
    short, _ = chat_cli.respond(loaded, messages, 3, io.StringIO(), temperature=0.0)
    assert history[-1]["content"] == short  # the reply was written with the new length
    assert short != chat_cli.respond(loaded, messages, 8, io.StringIO(), temperature=0.0)[0]

    chat_cli.chat_loop(loaded, args, scripted(["/settings top-k 40", "/settings reset"]), out)
    assert "Top-k is now 40" in out.getvalue() and "back to what this chat started with" in out.getvalue()
    assert (args.max_new_tokens, args.top_k) == (3, 0)  # what that chat started with


def test_chat_model_lists_and_switches(tiny_run, tmp_path, monkeypatch):
    """The tiny model is a base model; it is re-saved with ``stage: sft`` to stand in for a chat model."""
    state = torch.load(tiny_run.checkpoint, map_location="cpu", weights_only=True)
    run = tmp_path / "runs" / "chat"
    run.mkdir(parents=True)
    torch.save({**state, "stage": "sft", "step": 20, "best_val_loss": 3.5}, run / "best.pt")
    (tmp_path / "broken.pt").write_bytes(b"half-written")
    monkeypatch.chdir(tmp_path)  # /model looks in release/ and runs/ of the current folder
    out = io.StringIO()
    inputs = ["/model 2", "/info", "The cat", "/model", "/model 1", "/model 7", "/model nowhere.pt"]
    inputs += ["/model broken.pt", f'/model "{tiny_run.checkpoint}"', "/info", "The dog"]
    history = chat_cli.chat_loop(load_model(tiny_run.checkpoint), chat_args(), scripted(inputs), out)
    text = out.getvalue()
    first, second = text.split("Chat model, 20 steps", 1)  # /info after the first switch
    assert "Now talking to chat/best.pt." in first and "conversation so far" not in first
    assert "(in use)" in second and "chat · step 20 · best · val loss 3.500 (in use)" in second
    assert "Already talking to chat/best.pt" in second and "There is no model 7" in second
    assert "There is no model at nowhere.pt" in second
    assert "Couldn't load broken.pt" in second and "Still talking to chat/best.pt" in second
    back = second.split("Couldn't load", 1)[1]
    assert "Now talking to" in back and "It can see the conversation so far." in back
    assert "This is a Base model" in back and "Base model (continues text)" in back
    assert [m["role"] for m in history] == ["user", "assistant"] * 2  # both models answered


# ---------------------------------------------------------------- terminal look


def test_plain_output_has_no_escape_codes(tiny_run):
    out = io.StringIO()
    chat_cli.chat_loop(load_model(tiny_run.checkpoint), chat_args(), scripted(["The cat", "/help"]), out)
    assert "\x1b" not in out.getvalue()


def test_styled_chat_draws_the_welcome_panel_and_replies(tiny_run):
    out = io.StringIO()
    term = Term(out, fancy=True, color=True, unicode=True, light=False)
    inputs = ["The cat", "/help", "/info", "/new", "/exit"]
    history = chat_cli.chat_loop(load_model(tiny_run.checkpoint), chat_args(), scripted(inputs), out, term)
    text = out.getvalue()
    assert "What shall we learn today?" in text and "╭" in text and "█▄▖" in text  # panel and fox
    assert "\x1b[38;" in text  # Vulpes orange
    assert "●" in text and "tokens" in text  # a reply and its meta line
    assert "/retry" in text and "Trained as" in text and "New chat" in text
    assert history == []


def test_styled_chat_draws_models_and_settings(tiny_run, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no release/ or runs/ here: only the model in use is listed
    out = io.StringIO()
    term = Term(out, fancy=True, color=True, unicode=True, light=False)
    inputs = ["/model", "/settings", "/settings penalty 1.2"]
    chat_cli.chat_loop(load_model(tiny_run.checkpoint), chat_args(), scripted(inputs), out, term)
    text = ANSI.sub("", out.getvalue())
    assert "Models" in text and "(in use)" in text and "No other chat model" in text
    assert "Settings" in text and "Reply length" in text and "Repetition penalty is now 1.2" in text


def test_term_helpers():
    assert cell_width("abc") == 3 and cell_width("日本") == 4 and cell_width("é") == 1
    assert cell_width("\x1b[1mhi\x1b[0m") == 2  # colour codes take no room
    assert clip("Explain the water cycle", 12) == "Explain the…"
    assert clip("short", 12) == "short"
    assert wrap("one two three\n\nfour", 8) == ["one two", "three", "", "four"]
    plain = Term(io.StringIO())
    assert not plain.fancy and plain.paint("x", "brand", bold=True) == "x"
    assert Term(io.StringIO(), fancy=True, color=True).paint("x", "brand").startswith("\x1b[")


def write_reply(pieces, width, fancy=True):
    out = io.StringIO()
    writer = ReplyWriter(Term(out, fancy=fancy, color=False, unicode=True), width)
    for piece in pieces:
        writer.push(piece)
    return writer.finish(), out.getvalue()


def test_reply_writer_tidies_space_and_wraps_under_an_indent():
    wrote, text = write_reply(["\n  Hello", " world", "\n\n\n\n", "foo  ", "\n\n"], width=40)
    assert wrote and text == "● Hello world\n\n  foo\n"  # no leading space, one blank line, no trailing ones
    _, text = write_reply(["aaaa bbbb cccc"], width=12)
    assert text == "● aaaa bbbb\n  cccc\n"
    _, text = write_reply(["aaaa bb", "bbbb"], width=10)  # "bbbbbb" reaches the edge part-way
    assert text == "● aaaa bbb\x1b[3D\x1b[K\n  bbbbbb\n"  # the cursor goes back and the word moves down
    assert write_reply(["  ", "\n"], width=10) == (False, "")
