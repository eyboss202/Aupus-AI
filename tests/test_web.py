"""Tests for the browser chat: settings and message checks, the HTTP API, streaming and saving."""

from __future__ import annotations

import http.client
import json
import threading
from pathlib import Path

import pytest
import torch

from aupus.cli import web
from aupus.data.sft import identity_conversations
from aupus.model import generate
from aupus.model.io import export_release, load_model
from aupus.tokenizer import END, EOS
from aupus.tokenizer.chat import prompt_ids


@pytest.fixture(scope="module")
def loaded(tiny_run):
    return load_model(tiny_run.checkpoint)


@pytest.fixture
def server(loaded, tmp_path):
    srv = web.ChatServer(("127.0.0.1", 0), loaded, tmp_path / "chats", seed=0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    srv.server_close()


def request(srv, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=30)
    data = None if body is None else json.dumps(body).encode()
    hdrs = {"Content-Type": "application/json", **(headers or {})}
    conn.request(method, path, body=data, headers=hdrs)
    res = conn.getresponse()
    payload = res.read().decode("utf-8")
    conn.close()
    return res.status, res.getheader("Content-Type", ""), payload


def events_of(payload):
    return [
        json.loads(chunk[len("data: ") :]) for chunk in payload.split("\n\n") if chunk.startswith("data: ")
    ]


GREEDY = {"temperature": 0, "repetition_penalty": 1.0, "max_new_tokens": 8}


# ---------------------------------------------------------------- pure functions


def test_parse_settings_clamps_and_rejects():
    s = web.parse_settings({"temperature": 9, "top_k": 3.7, "max_new_tokens": 0})
    assert s["temperature"] == 2.0 and s["top_k"] == 3 and s["max_new_tokens"] == 1
    assert web.parse_settings(None) == web.DEFAULTS
    for bad in (
        {"temperature": "hot"},
        {"temperature": True},
        {"temperature": float("nan")},
        {"seed": 1},
        [],
    ):
        with pytest.raises(web.RequestError):
            web.parse_settings(bad)


def test_parse_messages():
    ok = [{"role": "user", "content": "hi"}]
    assert web.parse_messages(ok, for_generation=True) == ok
    for bad in ("hi", [1], [{"role": "assistant", "content": "x"}], [{"role": "user", "content": 3}]):
        with pytest.raises(web.RequestError):
            web.parse_messages(bad, for_generation=True)
    with pytest.raises(web.RequestError, match="end with an assistant"):
        web.parse_messages(ok, for_generation=False)  # saving needs a finished exchange


def test_reply_events_match_greedy_generation(loaded):
    messages = [{"role": "user", "content": "The cat"}]
    events = list(web.reply_events(loaded, messages, web.parse_settings(GREEDY)))
    assert events[-1]["type"] == "done"
    tok = loaded.tokenizer
    stops = [tok.special_id(END), tok.special_id(EOS)]
    ids = generate(loaded.model, prompt_ids(tok, messages), 8, temperature=0, stop_ids=stops)
    assert "".join(e["text"] for e in events if e["type"] == "token") == tok.decode(ids, skip_special=True)
    assert events[-1]["reply"] == tok.decode(ids, skip_special=True).strip()


def test_done_event_says_why_the_reply_ended_and_counts_its_tokens(loaded, monkeypatch):
    """The page shows a reply that ran out of room as possibly cut off, and how full the memory is."""
    tok = loaded.tokenizer
    messages = [{"role": "user", "content": "The cat"}]
    word = tok.encode(" sat")

    def reply_with(ids):
        def fake_stream(*args, **kwargs):
            yield from ids

        monkeypatch.setattr(web, "stream", fake_stream)
        return list(web.reply_events(loaded, messages, web.parse_settings(GREEDY)))[-1]

    prompt = len(prompt_ids(tok, messages))
    ended = reply_with([*word, tok.special_id(END)])
    assert ended["reply"] == "sat" and ended["finish"] == "stop"
    assert ended["usage"] == {
        "prompt_tokens": prompt,
        "completion_tokens": len(word) + 1,
        "context_length": loaded.model.cfg.context_length,
    }
    cut = reply_with(word)  # the tokens ran out before the model ended its turn
    assert cut["finish"] == "length" and cut["usage"]["completion_tokens"] == len(word)


def test_reply_events_report_dropped_history_and_overlong_messages(loaded):
    long = "The cat and the dog " * 5
    messages = []
    for _ in range(4):
        messages += [{"role": "user", "content": long}, {"role": "assistant", "content": long}]
    messages.append({"role": "user", "content": "The cat"})
    first = next(web.reply_events(loaded, messages, web.parse_settings(GREEDY)))
    assert first["type"] == "dropped" and first["count"] > 0  # context window is only 64 here
    with pytest.raises(web.RequestError, match="too long"):
        web.reply_events(loaded, [{"role": "user", "content": long * 20}], web.parse_settings(GREEDY))


def test_stopping_early_restores_the_model(loaded):
    events = web.reply_events(loaded, [{"role": "user", "content": "The cat"}], web.parse_settings(None))
    next(events, None)
    events.close()
    assert not loaded.model.training  # inference model stays in eval mode, generation released


def test_saved_chat_is_valid_identity_format(tmp_path):
    convo = [{"role": "user", "content": "Who are you?"}, {"role": "assistant", "content": "I'm Aupus."}]
    first = web.save_conversation(convo, tmp_path)
    second = web.save_conversation(convo, tmp_path)
    assert first != second and first.name.startswith("chat-")
    (conv,) = identity_conversations(first)  # can go straight into the fine-tuning data
    assert conv["messages"] == convo


# ---------------------------------------------------------------- HTTP API


def test_page_and_info(server):
    status, ctype, page = request(server, "GET", "/")
    assert status == 200 and ctype.startswith("text/html") and "<title>Aupus Chat</title>" in page
    # What the model writes only ever reaches the page as text: no way to build HTML from a string.
    assert not any(
        sink in page for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write")
    )
    status, _, info = request(server, "GET", "/api/info")
    info = json.loads(info)
    assert status == 200 and info["stage"] == "pretrain" and info["context_length"] == 64
    assert info["defaults"] == web.DEFAULTS
    assert request(server, "GET", "/nope")[0] == 404
    status, ctype, logo = request(server, "GET", "/logo.svg")
    assert status == 200 and ctype == "image/svg+xml" and logo.lstrip().startswith("<svg")


def test_fonts_are_bundled_and_served_by_exact_name(server):
    """The page's typefaces work offline; only the bundled files can be fetched, nothing else on disk."""
    for name in ("newsreader-latin-standard-normal.woff2", "instrument-sans-latin-standard-normal.woff2"):
        conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
        conn.request("GET", f"/fonts/{name}")
        res = conn.getresponse()
        body = res.read()
        conn.close()
        assert res.status == 200 and res.getheader("Content-Type") == "font/woff2" and body[:4] == b"wOF2"
    for path in ("/fonts/../web.py", "/fonts/OFL-newsreader.txt", "/fonts/nope.woff2"):
        assert request(server, "GET", path)[0] == 404
    fonts = Path(web.__file__).parent / "fonts"
    assert {"OFL-newsreader.txt", "OFL-instrument-sans.txt"} <= {f.name for f in fonts.iterdir()}


def test_packaged_logo_matches_the_docs_logo():
    """The tab icon is a copy of docs/assets/logo-small.svg (the package can't reach docs/)."""
    root = Path(__file__).resolve().parents[1]
    docs = (root / "docs" / "assets" / "logo-small.svg").read_text(encoding="utf-8")
    packaged = (root / "aupus" / "cli" / "logo.svg").read_text(encoding="utf-8")
    assert packaged == docs


def test_chat_streams_the_same_reply_as_the_library(server, loaded):
    messages = [{"role": "user", "content": "The cat"}]
    status, ctype, payload = request(server, "POST", "/api/chat", {"messages": messages, "settings": GREEDY})
    assert status == 200 and ctype.startswith("text/event-stream")
    events = events_of(payload)
    expected = list(web.reply_events(loaded, messages, web.parse_settings(GREEDY)))
    assert events == expected


def test_an_error_part_way_is_reported_to_the_page(server, monkeypatch, capsys):
    """Once the stream has started there is no status code left to send: the page gets an ``error`` event."""

    def broken(*args, **kwargs):
        yield {"type": "token", "text": "Half a"}
        raise RuntimeError("out of memory")

    msg = {"messages": [{"role": "user", "content": "The cat"}], "settings": GREEDY}
    monkeypatch.setattr(web, "reply_events", broken)
    status, _, payload = request(server, "POST", "/api/chat", msg)
    events = events_of(payload)
    assert status == 200 and events[0] == {"type": "token", "text": "Half a"}
    assert events[-1]["type"] == "error" and "out of memory" in events[-1]["message"]
    assert "out of memory" in capsys.readouterr().err  # and the terminal shows it too
    monkeypatch.undo()
    status, _, payload = request(server, "POST", "/api/chat", msg)  # the server carries on
    assert status == 200 and events_of(payload)[-1]["type"] == "done"


def test_chat_rejects_bad_requests(server):
    assert request(server, "POST", "/api/chat", {"messages": "hi"})[0] == 400
    assert (
        request(
            server,
            "POST",
            "/api/chat",
            {"messages": [{"role": "user", "content": "x"}], "settings": {"x": 1}},
        )[0]
        == 400
    )
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
    conn.request("POST", "/api/chat", body=b"{not json", headers={"Content-Type": "application/json"})
    assert conn.getresponse().status == 400
    conn.close()


def test_one_reply_at_a_time(server):
    server.generating.acquire()
    try:
        status, _, body = request(
            server, "POST", "/api/chat", {"messages": [{"role": "user", "content": "x"}]}
        )
        assert status == 409 and "still being written" in body
    finally:
        server.generating.release()


def test_other_sites_are_refused(server):
    """A page on another site (or a DNS-rebinding hostname) must not be able to use the model."""
    msg = {"messages": [{"role": "user", "content": "x"}]}
    assert request(server, "GET", "/api/info", headers={"Host": "evil.example:80"})[0] == 403
    assert request(server, "POST", "/api/chat", msg, headers={"Origin": "http://evil.example"})[0] == 403
    ok_origin = {"Origin": f"http://127.0.0.1:{server.port}"}
    assert request(server, "POST", "/api/chat", {**msg, "settings": GREEDY}, headers=ok_origin)[0] == 200


def test_save_endpoint(server):
    convo = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello."}]
    status, _, body = request(server, "POST", "/api/save", {"messages": convo})
    path = Path(json.loads(body)["path"])
    assert status == 200 and json.loads(path.read_text(encoding="utf-8"))["messages"] == convo
    assert request(server, "POST", "/api/save", {"messages": convo[:1]})[0] == 400


# ---------------------------------------------------------------- model picker


@pytest.fixture(scope="module")
def project(tiny_run, tmp_path_factory):
    """A project folder like the real one: a chat run (one checkpoint released) and a base run.

    The tiny model is a base model; it is re-saved with ``stage: sft`` to stand in for a chat model.
    """
    root = tmp_path_factory.mktemp("project")
    state = torch.load(tiny_run.checkpoint, map_location="cpu", weights_only=True)
    chat = root / "runs" / "chat"
    chat.mkdir(parents=True)
    torch.save({**state, "stage": "sft", "step": 20, "best_val_loss": 3.5}, chat / "best.pt")
    torch.save({**state, "stage": "sft", "step": 20}, chat / "step_000020.pt")  # the same step as best.pt
    torch.save({**state, "stage": "sft", "step": 40}, chat / "step_000040.pt")
    (chat / "metrics.jsonl").write_text(
        '{"step": 20, "val/loss": 3.5}\nnot json\n{"step": 40, "val/loss": 3.25}\n'
    )
    (chat / "step_000060.pt").write_bytes(b"half-written")  # training was interrupted: skipped
    base = root / "runs" / "base"
    base.mkdir()
    torch.save(state, base / "best.pt")  # a base model: it doesn't chat, so it isn't offered
    export_release(chat / "best.pt", root / "release" / "chat-0.1")
    export_release(base / "best.pt", root / "release" / "base-0.1")
    return root


def test_find_models_lists_chat_models_once(project):
    models = web.find_models(project)
    assert [m.id for m in models] == ["release/chat-0.1", "runs/chat/step_000040.pt"]
    release, step40 = models
    assert release.label == "chat-0.1 · step 20 · release" and release.val_loss == 3.5
    assert step40.label == "chat · step 40" and step40.val_loss == 3.25  # from metrics.jsonl
    assert web.find_models(project / "nothing-here") == []


@pytest.fixture
def picker(project, tmp_path):
    models = web.find_models(project)
    srv = web.ChatServer(("127.0.0.1", 0), load_model(models[0].path), tmp_path / "chats", 0, models)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    srv.server_close()


def test_info_lists_the_models(picker):
    info = json.loads(request(picker, "GET", "/api/info")[2])
    assert info["model"] == "release/chat-0.1" and info["stage"] == "sft"
    assert [m["id"] for m in info["models"]] == ["release/chat-0.1", "runs/chat/step_000040.pt"]
    assert info["models"][1] == {
        "id": "runs/chat/step_000040.pt",
        "label": "chat · step 40",
        "step": 40,
        "val_loss": 3.25,
    }


def test_switching_models(picker):
    status, _, body = request(picker, "POST", "/api/model", {"model": "runs/chat/step_000040.pt"})
    info = json.loads(body)
    assert status == 200 and info["model"] == "runs/chat/step_000040.pt" and info["step"] == 40
    assert picker.loaded.info["step"] == 40 and not picker.loaded.model.training
    msg = {"messages": [{"role": "user", "content": "The cat"}], "settings": GREEDY}
    status, _, payload = request(picker, "POST", "/api/chat", msg)
    assert status == 200 and events_of(payload)[-1]["type"] == "done"  # the new model answers
    assert request(picker, "POST", "/api/model", {"model": "release/chat-0.1"})[0] == 200
    assert picker.model_id == "release/chat-0.1"


def test_only_listed_models_can_be_loaded(picker, project):
    """The page picks from the list; it can't make the server open other files."""
    for bad in ("runs/base/best.pt", str(project / "runs" / "base" / "best.pt"), "../x", 3, None):
        status, _, body = request(picker, "POST", "/api/model", {"model": bad})
        assert status == 400 and "unknown model" in body
    assert picker.model_id == "release/chat-0.1"


def test_no_switching_while_a_reply_is_written(picker):
    picker.generating.acquire()
    try:
        status, _, body = request(picker, "POST", "/api/model", {"model": "runs/chat/step_000040.pt"})
        assert status == 409 and "still being written" in body
        picker.switching = True
        status, _, body = request(
            picker, "POST", "/api/chat", {"messages": [{"role": "user", "content": "x"}]}
        )
        assert status == 409 and "model is loading" in body
    finally:
        picker.switching = False
        picker.generating.release()
    assert picker.model_id == "release/chat-0.1"


def test_a_broken_model_file_keeps_the_current_model(picker, project, tmp_path):
    broken = tmp_path / "broken.pt"
    broken.write_bytes(b"not a checkpoint")
    picker.models["broken"] = web.ModelChoice("broken", broken, "broken")
    status, _, body = request(picker, "POST", "/api/model", {"model": "broken"})
    assert status == 500 and "couldn't load broken" in body
    assert picker.model_id == "release/chat-0.1" and picker.loaded.info["step"] == 20


def test_the_starting_model_is_always_offered(loaded, tmp_path):
    """Started with a model the scan doesn't list (here a base model), it is still in the list, first."""
    other = web.ModelChoice("release/x", tmp_path / "x", "x")
    srv = web.ChatServer(("127.0.0.1", 0), loaded, tmp_path / "chats", 0, [other])
    try:
        assert list(srv.models) == [loaded.source.as_posix(), "release/x"]
        assert srv.model_id == loaded.source.as_posix()
        assert srv.models[srv.model_id].label.endswith("· best · base")
    finally:
        srv.server_close()


def test_a_second_server_cannot_take_a_port_in_use(server, loaded, tmp_path):
    """Two chat servers on one port would split the requests between them (Windows allowed it)."""
    with pytest.raises(OSError):
        web.ChatServer(("127.0.0.1", server.port), loaded, tmp_path / "chats").server_close()
    assert request(server, "GET", "/api/info")[0] == 200  # the first one is unaffected
