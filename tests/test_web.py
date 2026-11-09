"""Tests for the browser chat: settings and message checks, the HTTP API, streaming and saving."""

from __future__ import annotations

import http.client
import json
import threading
from pathlib import Path

import pytest

from aupus.cli import web
from aupus.data.sft import identity_conversations
from aupus.model import generate
from aupus.model.io import load_model
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
