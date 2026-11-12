"""``chat``: talk to Aupus in the terminal (Milestone 11).

    python scripts/chat.py --model release/aupus-0.1-instruct
    python scripts/chat.py --model runs/aupus-0.1-instruct/best.pt --temperature 0.5

Type a message and press Enter. Commands start with ``/``: ``/help`` lists them (``/new`` starts
over, ``/retry`` writes the last reply again, ``/save`` keeps the chat for training, ``/exit``
quits; so does Ctrl-D). Ctrl-C during a reply stops that reply.

**Look.** In a terminal the chat has the browser chat's colours and owl (``aupus/cli/term.py``): a
welcome panel, a ruled input line with a status line under it (commands, and how much of the
model's memory the chat fills), and replies that stream in word-wrapped. Piped or redirected,
it writes plain text. ``NO_COLOR=1`` turns the colours off; ``AUPUS_THEME=light`` suits a light
terminal background.

**Memory.** The model has no memory between calls. Each reply is generated from
the whole conversation so far, laid out in the chat template
(``aupus/tokenizer/chat.py``). When that no longer fits the context window
(1,024 tokens for Aupus 0.1, minus room for the reply), the oldest exchanges are
dropped first. So in a long chat Aupus forgets the beginning.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import random
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from aupus import __version__
from aupus.cli.generate import add_model_args, add_sampling_args, open_model, sampling_kwargs
from aupus.cli.streaming import TextStream
from aupus.cli.term import (
    ANSI,
    ReplyWriter,
    Spinner,
    Term,
    cell_width,
    clip,
    number,
    pad,
    run_with_spinner,
    wrap,
)
from aupus.model import stream
from aupus.model.io import LoadedModel
from aupus.tokenizer import END, EOS, Tokenizer
from aupus.tokenizer.chat import Message, prompt_ids

COMMANDS: list[tuple[str, str]] = [
    ("/help", "show these commands"),
    ("/new", "start over: Aupus forgets the conversation (also /reset, /clear)"),
    ("/retry", "write the last reply again"),
    ("/save", "save this chat to correct by hand and train on"),
    ("/info", "the model and the sampling settings"),
    ("/exit", "quit (also /quit, Ctrl-D)"),
]
HELP = "Commands: " + ", ".join(c for c, _ in COMMANDS) + ". Ctrl-C stops a reply."
SUGGESTIONS = [
    "What is a prime number?",
    "Why is the sky blue?",
    "Give me three tips for learning to read.",
    "Explain the water cycle simply.",
]
#: What the spinner says while Aupus reads the conversation, before its first word.
THINKING = ["Pondering", "Hooting", "Mulling it over", "Recalling", "Perching", "Musing", "Preening"]
PLAIN_PROMPT = "\nyou> "


@dataclass
class Reply:
    """One generated reply and how it went."""

    text: str = ""
    dropped: int = 0  # old messages left out so the conversation fits the context window
    prompt_tokens: int = 0
    tokens: int = 0  # generated, special tokens included
    seconds: float = 0.0  # from the first generated token to the last
    finish: str = "stop"  # "stop": the model ended it; "length": out of room; "stopped": Ctrl-C


def fit_history(
    tokenizer: Tokenizer, messages: list[Message], window: int, reserve: int
) -> tuple[list[int], list[Message]]:
    """Prompt ids for ``messages``; drops the oldest user/assistant pairs until ``reserve`` ids are free.

    The system message (if any) and the newest user message are always kept.
    Raises ``ValueError`` if even those don't fit.
    """
    system = messages[:1] if messages and messages[0]["role"] == "system" else []
    turns = messages[len(system) :]
    while True:
        ids = prompt_ids(tokenizer, system + turns)
        if len(ids) + reserve <= window:
            return ids, system + turns
        if len(turns) <= 1:
            raise ValueError(
                f"the message is too long: {len(ids)} tokens, and the context window is {window} "
                f"(with {reserve} kept free for the reply)"
            )
        turns = turns[2:]


def generate_reply(
    loaded: LoadedModel,
    messages: list[Message],
    max_new_tokens: int,
    write: Callable[[str], None],
    **sampling: object,
) -> Reply:
    """Generate the assistant's next reply, passing each piece of text to ``write`` as it comes.

    Raises ``ValueError`` (before generating) if even the newest message doesn't fit. Ctrl-C ends
    the reply early (``finish == "stopped"``); the text so far is kept.
    """
    tok = loaded.tokenizer
    window = loaded.model.cfg.context_length
    reserve = min(max_new_tokens, window // 2)
    ids, kept = fit_history(tok, messages, window, reserve)
    reply = Reply(dropped=len(messages) - len(kept), prompt_tokens=len(ids))
    text = TextStream(tok)
    stops = [tok.special_id(END), tok.special_id(EOS)]
    first = last = None
    try:
        tokens = stream(loaded.model, ids, reserve, stop_ids=stops, vocab_limit=tok.vocab_size, **sampling)  # type: ignore[arg-type]
        for token in tokens:
            first = first or time.perf_counter()
            reply.tokens, last = reply.tokens + 1, token
            write(text.push(token))
        reply.finish = "stop" if last in stops else "length"
    except KeyboardInterrupt:
        reply.finish = "stopped"
    write(text.flush())
    if first and reply.tokens > 1:
        reply.seconds = time.perf_counter() - first
    reply.text = text.text.strip()
    return reply


def respond(
    loaded: LoadedModel, messages: list[Message], max_new_tokens: int, out: TextIO, **sampling: object
) -> tuple[str, int]:
    """Stream the assistant's next reply to ``out`` as plain text.

    Returns ``(reply, messages dropped to fit)``.
    """

    def write(piece: str) -> None:
        out.write(piece)
        out.flush()

    reply = generate_reply(loaded, messages, max_new_tokens, write, **sampling)
    if reply.finish == "stopped":
        out.write(" [stopped]")
    return reply.text, reply.dropped


# ---------------------------------------------------------------- the screen


def model_name(loaded: LoadedModel) -> str:
    """``aupus-0.1-instruct`` for a release, ``aupus-0.1-instruct/best.pt`` for a checkpoint."""
    source = loaded.source
    return source.name if source.is_dir() else f"{source.parent.name}/{source.name}"


def is_base(loaded: LoadedModel) -> bool:
    return loaded.info.get("stage", "pretrain") == "pretrain"


class ChatView:
    """Draws the chat: plain text when ``term`` isn't fancy (pipes, tests), styled otherwise."""

    def __init__(self, term: Term, loaded: LoadedModel, max_new_tokens: int) -> None:
        self.t, self.loaded, self.max_new_tokens = term, loaded, max_new_tokens
        self.device = next(loaded.model.parameters()).device
        self.window = loaded.model.cfg.context_length
        self.used: int | None = None  # tokens of memory the chat filled at the last reply

    # ------------------------------------------------------------ welcome

    def welcome(self) -> None:
        t, loaded = self.t, self.loaded
        params = loaded.model.num_params() / 1e6
        dot = f" {t.g['dot']} "
        facts = f"{params:.0f}M params{dot}{self.device}{dot}{number(self.window)}-token memory"
        if not t.fancy:
            t.write(f"Aupus | {loaded.source} | {params:.0f}M params | {self.device}\n{HELP}\n")
            if is_base(loaded):
                t.write("Note: this is a Base model (not fine-tuned for chat); expect it to ramble.\n")
            return
        width = t.width()
        version = ".".join(__version__.split(".")[:2])  # "0.1", as the browser chat says it
        title = t.paint("Aupus", "brand", bold=True) + " " + t.paint(version, "muted")
        name = model_name(loaded)
        if loaded.info.get("step"):
            name += f"{dot}step {number(loaded.info['step'])}"
        left = [
            "",
            t.paint("What shall we learn today?", bold=True),
            "",
            *(t.paint(row, "brand") for row in t.owl()),
            "",
        ]
        if width >= 76:  # two columns: welcome and owl | tips and suggestions
            lw = 40
            rw = width - 4 - lw - 3
            left += [t.paint(clip(s, lw, t.g["more"]), "muted") for s in (name, facts)]
            right = [
                t.paint("Tips for getting started", "brand", bold=True),
                "Ask anything, then press Enter",
                f"{t.paint('/help', bold=True)} lists the commands",
                f"{t.paint('Ctrl-C', bold=True)} stops a reply",
                t.paint(t.g["h"] * rw, "rule"),
                t.paint("Try asking", "brand", bold=True),
                *(t.paint(clip(s, rw, t.g["more"]), "muted") for s in SUGGESTIONS),
            ]
            right = [""] + right
            height = max(len(left), len(right))
            left += [""] * (height - len(left))
            right += [""] * (height - len(right))
            rows = [
                pad(a, lw, "center") + " " + t.paint(t.g["v"], "rule") + " " + pad(b, rw)
                for a, b in zip(left, right, strict=True)
            ]
        else:  # one column
            inner = width - 4
            rows = [pad(r, inner, "center") for r in left]
            more = t.g["more"]
            rows += [pad(t.paint(clip(s, inner, more), "muted"), inner, "center") for s in (name, facts)]
            rows += ["", pad(f"{t.paint('/help', bold=True)} for commands", inner, "center"), ""]
        t.write("\n" + "\n".join(t.panel(rows, width, title)) + "\n")
        about = "Aupus 0.1 is small, built from scratch and still learning: it can be wrong."
        for line in wrap(about, width - 2):
            t.write("  " + t.paint(line, "muted") + "\n")
        if is_base(loaded):
            note = "This is a Base model: it learned to continue web text, not to chat, so it will ramble."
            for i, line in enumerate(wrap(note, width - 4)):
                t.write(t.paint(f"  {t.g['warn'] if i == 0 else ' '} {line}\n", "warn"))

    # ------------------------------------------------------------ input

    def status(self) -> str:
        t = self.t
        left = t.paint(f"  /help for commands {t.g['dot']} Ctrl-C stops a reply", "muted")
        if self.used is None:
            right = t.paint(model_name(self.loaded), "muted")
        else:
            reserve = min(self.max_new_tokens, self.window // 2)
            right = t.meter(self.used, self.window, full=self.used + reserve >= self.window)
        gap = t.width() - cell_width(left) - cell_width(right)
        return left + " " * gap + right if gap >= 2 else left

    def ask(self, read_line: Callable[[str], str]) -> str:
        """Read the next message. Raises ``EOFError`` / ``KeyboardInterrupt`` to end the chat."""
        t = self.t
        if not t.fancy:
            return read_line(PLAIN_PROMPT)
        prompt = t.paint(t.g["prompt"], "brand", bold=True) + " "
        # A ruled box: rule, the input line, rule, status. The cursor then goes back up to the input line.
        t.write("\n" + t.rule() + "\n\n" + t.rule() + "\n" + self.status() + t.up(2))
        try:
            if "readline" in sys.modules:  # readline redraws the prompt itself: mark the colour codes for it
                line = read_line(ANSI.sub(lambda m: f"\x01{m.group()}\x02", prompt))
            else:
                t.write(prompt)
                line = read_line("")
        except (EOFError, KeyboardInterrupt) as e:
            # Clear the box away. Windows' Ctrl-Z ends with Enter, which moved the cursor down a line.
            moved = 2 if isinstance(e, EOFError) and sys.platform == "win32" else 1
            t.write(t.up(moved) + t.clear_below())
            raise
        # Up to the first rule and clear the box away; the message is printed again as part of the chat.
        typed = max(1, math.ceil((2 + cell_width(line)) / t.columns()))
        t.write(t.up(typed + 1) + t.clear_below())
        return line

    def echo(self, line: str) -> None:
        """The user's message, as it stays in the conversation."""
        t = self.t
        if not t.fancy:
            return
        rows = wrap(line, t.width() - 2)
        t.write(t.paint(t.g["prompt"], "muted") + " " + t.paint(rows[0], "muted") + "\n")
        for row in rows[1:]:
            t.write("  " + t.paint(row, "muted") + "\n")

    def note(self, text: str, color: str = "muted") -> None:
        """A short answer to a command, or a warning: ``  └ text``."""
        t = self.t
        if not t.fancy:
            t.write(text + "\n")
            return
        for i, row in enumerate(wrap(text, t.width() - 5)):
            lead = t.g["elbow"] if i == 0 else " " * len(t.g["elbow"])
            t.write("  " + t.paint(lead, "rule") + " " + t.paint(row, color) + "\n")

    def table(self, title: str, rows: list[tuple[str, str]]) -> None:
        t = self.t
        if not t.fancy:
            t.write(title + "\n" + "".join(f"  {a:<14} {b}\n" for a, b in rows))
            return
        self.note(title, "brand")
        key_w = max(cell_width(a) for a, _ in rows) + 2
        for a, b in rows:
            lines = wrap(b, max(10, t.width() - 5 - key_w))
            t.write("     " + pad(t.paint(a, bold=True), key_w) + t.paint(lines[0], "muted") + "\n")
            for more in lines[1:]:
                t.write("     " + " " * key_w + t.paint(more, "muted") + "\n")

    # ------------------------------------------------------------ replies

    def reply(self, generate: Callable[[Callable[[str], None]], Reply]) -> Reply:
        """Run ``generate(write)``: a spinner until the first words, then the reply as it streams."""
        t = self.t
        if not t.fancy:
            t.write("aupus> ")
            reply = generate(t.write)
            if reply.finish == "stopped":
                t.write(" [stopped]")
            t.write("\n")
            if reply.dropped:
                t.write(f"(the {reply.dropped} oldest messages no longer fit in the context window)\n")
            return reply
        t.write("\n")
        writer = ReplyWriter(t, t.width())
        spinner = Spinner(t, random.choice(THINKING))
        spinner.start()

        def write(piece: str) -> None:
            if piece.strip():
                spinner.stop()
            writer.push(piece)

        try:
            reply = generate(write)
        finally:
            spinner.stop()
        if not writer.finish():
            t.write(t.paint(t.g["bullet"], "brand") + " " + t.paint("(no reply)", "muted") + "\n")
        self.used = reply.prompt_tokens + reply.tokens
        self.meta(reply)
        return reply

    def meta(self, reply: Reply) -> None:
        """Under a reply: its length and speed, and anything to know about it."""
        t = self.t
        dot = f" {t.g['dot']} "
        facts = [f"{number(reply.tokens)} token{'' if reply.tokens == 1 else 's'}"]
        if reply.seconds > 0:
            facts.append(f"{number((reply.tokens - 1) / reply.seconds)} tok/s")
        flags = {"length": "may be cut off: it reached the reply length (-n)", "stopped": "stopped"}
        line = t.paint(dot.join(facts), "faint")
        flag = flags.get(reply.finish)
        if flag and cell_width(dot.join(facts) + dot + flag) <= t.width() - 2:
            line += t.paint(dot, "faint") + t.paint(flag, "warn")
        elif flag:  # too narrow for one line
            line += "\n  " + t.paint(clip(flag, t.width() - 2, t.g["more"]), "warn")
        t.write("  " + line + "\n")
        if reply.dropped:
            self.note(
                f"The {reply.dropped} oldest messages no longer fit in Aupus's {number(self.window)}-token "
                "memory, so this reply was written without them.",
                "warn",
            )

    def fresh(self) -> None:
        """A new chat: an empty screen (scrollback included) and the welcome panel again."""
        if self.t.fancy:
            self.t.write("\x1b[H\x1b[2J\x1b[3J")
            self.welcome()

    def goodbye(self) -> None:
        if self.t.fancy:
            self.t.write("\n" + self.t.paint("  Goodbye. Keep learning!", "muted") + "\n\n")
        else:
            self.t.write("\n")


# ---------------------------------------------------------------- the conversation


def chat_loop(
    loaded: LoadedModel,
    args: argparse.Namespace,
    read_line: Callable[[str], str] = input,
    out: TextIO = sys.stdout,
    term: Term | None = None,
) -> list[Message]:
    """Run the conversation until /exit or end of input. Returns the final history."""
    term = term or Term(out)
    view = ChatView(term, loaded, args.max_new_tokens)
    sampling = sampling_kwargs(args, view.device)
    system: list[Message] = [{"role": "system", "content": args.system}] if args.system else []
    messages = list(system)
    view.welcome()

    def answer() -> bool:
        """Write Aupus's reply to the last user message. False (and a note) if it can't fit."""
        try:
            reply = view.reply(
                lambda write: generate_reply(loaded, messages, args.max_new_tokens, write, **sampling)
            )
        except ValueError as e:
            view.note(f"Couldn't reply: {e}. Try a shorter message." if term.fancy else f"({e})", "danger")
            return False
        messages.append({"role": "assistant", "content": reply.text})
        return True

    try:
        while True:
            try:
                line = view.ask(read_line).strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                continue
            view.echo(line)
            command = line.split()[0].lower() if line.startswith("/") else None
            if command in ("/exit", "/quit"):
                break
            if command in ("/reset", "/new", "/clear"):
                messages = list(system)
                view.used = None
                view.fresh()
                view.note("New chat: Aupus forgot the last one." if term.fancy else "(conversation cleared)")
                continue
            if command == "/help":
                view.table("Commands", COMMANDS + [("Ctrl-C", "stop the reply being written")])
                continue
            if command == "/info":
                view.table("This model", info_rows(loaded, args, view))
                continue
            if command == "/save":
                save(messages, args, view)
                continue
            if command == "/retry":
                if not messages or messages[-1]["role"] != "assistant":
                    view.note("Nothing to retry yet: ask something first.")
                    continue
                previous = messages.pop()
                view.note("Writing the last reply again.")
                if not answer():
                    messages.append(previous)
                continue
            if command:
                view.note(f"unknown command {command}. {HELP}", "warn")
                continue
            messages.append({"role": "user", "content": line})
            if not answer():
                messages.pop()
    finally:
        term.cursor(True)
    view.goodbye()
    return messages


def info_rows(loaded: LoadedModel, args: argparse.Namespace, view: ChatView) -> list[tuple[str, str]]:
    info = loaded.info
    kind = "Base model (continues text)" if is_base(loaded) else "Chat model"
    if info.get("step") is not None:
        kind += f", {number(info['step'])} steps"
    sampling = (
        f"temperature {args.temperature}, top-p {args.top_p}, top-k {args.top_k or 'off'}, "
        f"repetition penalty {args.repetition_penalty}, up to {args.max_new_tokens} tokens a reply"
    )
    rows = [
        ("Model", loaded.source.as_posix()),
        ("Trained as", kind),
        ("Size", f"{loaded.model.num_params() / 1e6:.1f} million parameters"),
        ("Memory", f"{number(view.window)} tokens (the conversation and the reply together)"),
        ("Running on", str(view.device)),
        ("Sampling", sampling),
    ]
    if args.system:
        rows.append(("System", args.system))
    if args.seed is not None:
        rows.append(("Seed", str(args.seed)))
    return rows


def save(messages: list[Message], args: argparse.Namespace, view: ChatView) -> None:
    """Write the chat in the format of ``data/identity.jsonl`` (like the browser chat's Save)."""
    from aupus.cli.web import save_conversation  # here: web imports this module

    if not any(m["role"] == "assistant" for m in messages):
        view.note("Nothing to save yet.")
        return
    path = save_conversation(messages, Path(getattr(args, "save_dir", None) or "data/chats"))
    view.note(f"Saved to {path.as_posix()}. Correct the replies by hand before training on it.")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_model_args(parser)
    parser.add_argument("--system", default=None, help="optional system message")
    parser.add_argument("--save-dir", type=Path, default=Path("data/chats"), help="where /save writes")
    add_sampling_args(parser, temperature=0.7, max_new=256)
    args = parser.parse_args(argv)
    with contextlib.suppress(ImportError):  # readline: arrow keys and history in input() (not on Windows)
        import readline  # noqa: F401
    term = Term(sys.stdout)
    loaded = run_with_spinner(term, "Waking Aupus up", lambda: open_model(args))
    chat_loop(loaded, args, term=term)  # type: ignore[arg-type]
