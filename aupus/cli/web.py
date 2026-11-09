"""``chat_web``: chat with Aupus in the browser (Milestone 11).

    python scripts/chat_web.py --model release/aupus-0.1-instruct
    python scripts/chat_web.py --model runs/aupus-0.1-instruct/best.pt --port 8080

then open http://127.0.0.1:8000. It is the terminal ``chat`` with a page around it:
the same chat template, the same context-window handling (``fit_history``: the
oldest exchanges are dropped when the conversation no longer fits) and the same
sampling settings. Replies appear token by token.

**Design.** A small HTTP server from the standard library (no web framework, in
keeping with the project's minimal dependencies) and one static HTML page
(``chat_web.html``):

* ``GET /``: the page; ``GET /logo.svg``: the Aupus owl (tab icon); ``GET /fonts/<file>.woff2``:
  the page's typefaces, Newsreader and Instrument Sans (bundled, so it works offline);
  ``GET /api/info``: the model and the default settings;
* ``POST /api/chat``: ``{"messages": [...], "settings": {...}}`` -> a stream of
  server-sent events: ``token`` (a piece of text), ``dropped`` (how many old
  messages no longer fit), ``done`` (the full reply) or ``error``;
* ``POST /api/save``: writes the conversation to ``<save-dir>/chat-<time>.jsonl``
  in the format of ``data/identity.jsonl``, so it can be corrected by hand and
  added to the fine-tuning data. Nothing is ever saved automatically.

The page keeps the conversation; the server is stateless apart from the model. One
GPU generates one reply at a time, so a second request while a reply is being
written gets HTTP 409. Closing the request (the page's Stop button) stops generation.

**Security.** It listens on 127.0.0.1 only by default. Requests whose ``Host``
or ``Origin`` header names another site are refused, so a web page open in the
same browser can't drive the model through this server ("DNS rebinding"). The
page never inserts model output as HTML, only as text. ``--host 0.0.0.0`` makes
it reachable from other devices on your network, with no password: only do that
on a network you trust.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import threading
from collections.abc import Iterator, Sequence
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any

import torch

from aupus.cli.chat import fit_history
from aupus.cli.generate import add_model_args, open_model
from aupus.cli.streaming import TextStream
from aupus.model import stream
from aupus.model.io import LoadedModel
from aupus.tokenizer import END, EOS
from aupus.tokenizer.chat import ChatFormatError, Message, validate

MAX_BODY = 1 << 20  # 1 MB: far more than a 1,024-token conversation needs
LOOPBACK = ("127.0.0.1", "localhost", "::1")

#: Sampling defaults (as in the terminal chat) and the range each setting is clamped to.
DEFAULTS: dict[str, float] = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 50,
    "repetition_penalty": 1.1,
    "max_new_tokens": 256,
}
LIMITS: dict[str, tuple[float, float]] = {
    "temperature": (0.0, 2.0),
    "top_p": (0.05, 1.0),
    "top_k": (0, 1000),
    "repetition_penalty": (1.0, 2.0),
    "max_new_tokens": (1, 1024),
}
INTEGER_SETTINGS = ("top_k", "max_new_tokens")


class RequestError(ValueError):
    """A bad request: reported to the page as HTTP 400 with this message."""


def parse_settings(raw: Any) -> dict[str, float]:
    """Defaults, overridden by the numbers in ``raw`` and clamped to ``LIMITS``."""
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise RequestError("settings must be an object")
    unknown = sorted(set(raw) - set(DEFAULTS))
    if unknown:
        raise RequestError(f"unknown setting(s): {', '.join(unknown)}")
    out = dict(DEFAULTS)
    for key, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
            raise RequestError(f"{key} must be a number")
        lo, hi = LIMITS[key]
        value = min(max(value, lo), hi)
        out[key] = int(value) if key in INTEGER_SETTINGS else float(value)
    return out


def parse_messages(raw: Any, *, for_generation: bool) -> list[Message]:
    """Check the conversation's shape and roles (``aupus.tokenizer.chat.validate``)."""
    if not isinstance(raw, list):
        raise RequestError("messages must be a list")
    for m in raw:
        if not isinstance(m, dict):
            raise RequestError("each message must be an object")
    try:
        validate(raw, for_generation=for_generation)
    except ChatFormatError as e:
        raise RequestError(str(e)) from None
    return [{"role": m["role"], "content": m["content"]} for m in raw]


def reply_events(
    loaded: LoadedModel, messages: list[Message], settings: dict[str, float], seed: int | None = None
) -> Iterator[dict[str, Any]]:
    """Generate the next assistant reply as a sequence of events (see the module docstring).

    Raises ``RequestError`` before the first event if even the newest message doesn't fit.
    Closing the iterator early (the client went away) stops generation.
    """
    tok, model = loaded.tokenizer, loaded.model
    window = model.cfg.context_length
    reserve = min(int(settings["max_new_tokens"]), window // 2)
    try:
        ids, kept = fit_history(tok, messages, window, reserve)
    except ValueError as e:
        raise RequestError(str(e)) from None
    generator = None
    if seed is not None:
        generator = torch.Generator(device=next(model.parameters()).device).manual_seed(seed)
    tokens = stream(
        model,
        ids,
        reserve,
        temperature=settings["temperature"],
        top_k=int(settings["top_k"]) or None,
        top_p=settings["top_p"] if settings["top_p"] < 1 else None,
        repetition_penalty=settings["repetition_penalty"],
        stop_ids=[tok.special_id(END), tok.special_id(EOS)],
        generator=generator,
        vocab_limit=tok.vocab_size,
    )

    def events() -> Iterator[dict[str, Any]]:
        if len(kept) < len(messages):
            yield {"type": "dropped", "count": len(messages) - len(kept)}
        text = TextStream(tok)
        try:
            for token in tokens:
                piece = text.push(token)
                if piece:
                    yield {"type": "token", "text": piece}
        finally:
            tokens.close()  # stopped early: end generation now (and restore the model's mode)
        tail = text.flush()
        if tail:
            yield {"type": "token", "text": tail}
        yield {"type": "done", "reply": text.text.strip()}

    return events()


def save_conversation(messages: list[Message], save_dir: Path) -> Path:
    """Write one ``{"messages": [...]}`` line, the format of ``data/identity.jsonl``."""
    save_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = save_dir / f"chat-{stamp}.jsonl"
    n = 1
    while path.exists():  # two saves in the same second
        n += 1
        path = save_dir / f"chat-{stamp}-{n}.jsonl"
    path.write_text(json.dumps({"messages": messages}, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


class ChatServer(ThreadingHTTPServer):
    """Holds the model; one reply is generated at a time (``generating``)."""

    daemon_threads = True

    def __init__(
        self, address: tuple[str, int], loaded: LoadedModel, save_dir: Path, seed: int | None = None
    ) -> None:
        super().__init__(address, ChatHandler)
        self.loaded, self.save_dir, self.seed = loaded, save_dir, seed
        self.generating = threading.Lock()
        files = resources.files("aupus.cli")
        self.page = files.joinpath("chat_web.html").read_bytes()
        self.logo = files.joinpath("logo.svg").read_bytes()  # a copy of docs/assets/logo.svg
        # The page's two typefaces (SIL Open Font License, see fonts/OFL-*.txt), served by exact name
        # only: nothing from the request path ever reaches the filesystem.
        fonts = files.joinpath("fonts")
        self.fonts = {f.name: f.read_bytes() for f in fonts.iterdir() if f.name.endswith(".woff2")}

    @property
    def port(self) -> int:
        return self.server_address[1]

    def allowed_hosts(self) -> set[str] | None:
        """``Host`` values to accept, or None (any) when listening beyond this machine."""
        host = self.server_address[0]
        if host not in LOOPBACK:
            return None
        return {f"{h}:{self.port}" for h in ("127.0.0.1", "localhost", "[::1]")}


class ChatHandler(BaseHTTPRequestHandler):
    server: ChatServer
    server_version = "Aupus"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 (stdlib signature)
        pass  # one line per request would drown the terminal; errors are reported to the page

    # ------------------------------------------------------------ helpers

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, data: dict[str, Any]) -> None:
        self._send(status, json.dumps(data).encode("utf-8"), "application/json")

    def _origin_ok(self) -> bool:
        allowed = self.server.allowed_hosts()
        if allowed is None:
            return True
        if self.headers.get("Host") not in allowed:
            return False
        origin = self.headers.get("Origin")
        return origin is None or origin in {f"http://{h}" for h in allowed}

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            raise RequestError("Content-Length required") from None
        if not 0 < length <= MAX_BODY:
            raise RequestError(f"request body must be 1 byte to {MAX_BODY} bytes")
        try:
            data = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise RequestError("request body is not valid JSON") from None
        if not isinstance(data, dict):
            raise RequestError("request body must be a JSON object")
        return data

    # ------------------------------------------------------------ routes

    def do_GET(self) -> None:
        if not self._origin_ok():
            return self._json(HTTPStatus.FORBIDDEN, {"error": "forbidden host or origin"})
        path = self.path.split("?", 1)[0]
        if path == "/":
            return self._send(HTTPStatus.OK, self.server.page, "text/html; charset=utf-8")
        if path in ("/logo.svg", "/favicon.ico"):
            return self._send(HTTPStatus.OK, self.server.logo, "image/svg+xml")
        if path.startswith("/fonts/") and path[len("/fonts/") :] in self.server.fonts:
            return self._send(HTTPStatus.OK, self.server.fonts[path[len("/fonts/") :]], "font/woff2")
        if path == "/api/info":
            loaded = self.server.loaded
            return self._json(
                HTTPStatus.OK,
                {
                    "source": str(loaded.source),
                    "params_m": round(loaded.model.num_params() / 1e6, 1),
                    "context_length": loaded.model.cfg.context_length,
                    "device": str(next(loaded.model.parameters()).device),
                    "stage": loaded.info.get("stage", "pretrain"),
                    "step": loaded.info.get("step"),
                    "defaults": DEFAULTS,
                    "limits": LIMITS,
                },
            )
        return self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:
        if not self._origin_ok():
            return self._json(HTTPStatus.FORBIDDEN, {"error": "forbidden host or origin"})
        path = self.path.split("?", 1)[0]
        try:
            if path == "/api/chat":
                return self._chat()
            if path == "/api/save":
                return self._save()
        except RequestError as e:
            return self._json(HTTPStatus.BAD_REQUEST, {"error": str(e)})
        return self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def _save(self) -> None:
        messages = parse_messages(self._read_json().get("messages"), for_generation=False)
        path = save_conversation(messages, self.server.save_dir)
        self._json(HTTPStatus.OK, {"path": path.as_posix()})

    def _chat(self) -> None:
        body = self._read_json()
        messages = parse_messages(body.get("messages"), for_generation=True)
        settings = parse_settings(body.get("settings"))
        if not self.server.generating.acquire(blocking=False):
            return self._json(HTTPStatus.CONFLICT, {"error": "another reply is still being written"})
        try:
            events = reply_events(self.server.loaded, messages, settings, self.server.seed)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()  # HTTP/1.0: the connection closes after the stream, no length needed
            try:
                for event in events:
                    self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass  # the page pressed Stop or was closed: stop generating
            finally:
                events.close()
        finally:
            self.server.generating.release()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_model_args(parser)
    parser.add_argument("--host", default="127.0.0.1", help="127.0.0.1 = this computer only (default)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--save-dir", type=Path, default=Path("data/chats"), help="where 'Save chat' writes")
    parser.add_argument("--seed", type=int, default=None, help="fix the dice for reproducible replies")
    args = parser.parse_args(argv)

    loaded = open_model(args)
    server = ChatServer((args.host, args.port), loaded, args.save_dir, args.seed)
    host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    params = loaded.model.num_params() / 1e6
    print(f"Aupus | {loaded.source} | {params:.0f}M params | {next(loaded.model.parameters()).device}")
    if args.host not in LOOPBACK:
        print(f"warning: listening on {args.host}: anyone on your network can use it (no password)")
    print(f"open http://{host}:{server.port} in your browser (Ctrl-C to quit)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        server.server_close()
