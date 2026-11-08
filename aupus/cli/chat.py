"""``chat``: talk to Aupus in the terminal (Milestone 11).

    python scripts/chat.py --model release/aupus-0.1-instruct
    python scripts/chat.py --model runs/aupus-0.1-instruct/best.pt --temperature 0.5

Commands inside the chat: ``/reset`` forgets the conversation, ``/exit`` (or
Ctrl-D) quits. Ctrl-C during a reply stops that reply.

**Memory.** The model has no memory between calls. Each reply is generated from
the whole conversation so far, laid out in the chat template
(``aupus/tokenizer/chat.py``). When that no longer fits the context window
(1,024 tokens for Aupus 0.1, minus room for the reply), the oldest exchanges are
dropped first. So in a long chat Aupus forgets the beginning.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from collections.abc import Callable, Sequence
from typing import TextIO

from aupus.cli.generate import add_model_args, add_sampling_args, open_model, sampling_kwargs
from aupus.cli.streaming import TextStream
from aupus.model import stream
from aupus.model.io import LoadedModel
from aupus.tokenizer import END, EOS, Tokenizer
from aupus.tokenizer.chat import Message, prompt_ids

HELP = "Commands: /reset (forget the conversation), /exit (quit). Ctrl-C stops a reply."


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


def respond(
    loaded: LoadedModel, messages: list[Message], max_new_tokens: int, out: TextIO, **sampling: object
) -> tuple[str, int]:
    """Stream the assistant's next reply to ``out``. Returns ``(reply, messages dropped to fit)``."""
    tok = loaded.tokenizer
    window = loaded.model.cfg.context_length
    reserve = min(max_new_tokens, window // 2)
    ids, kept = fit_history(tok, messages, window, reserve)
    text = TextStream(tok)
    stops = [tok.special_id(END), tok.special_id(EOS)]
    try:
        for token in stream(loaded.model, ids, reserve, stop_ids=stops, **sampling):  # type: ignore[arg-type]
            out.write(text.push(token))
            out.flush()
    except KeyboardInterrupt:
        out.write(" [stopped]")
    out.write(text.flush())
    return text.text.strip(), len(messages) - len(kept)


def chat_loop(
    loaded: LoadedModel,
    args: argparse.Namespace,
    read_line: Callable[[str], str] = input,
    out: TextIO = sys.stdout,
) -> list[Message]:
    """Run the conversation until /exit or end of input. Returns the final history."""
    device = next(loaded.model.parameters()).device
    sampling = sampling_kwargs(args, device)
    system: list[Message] = [{"role": "system", "content": args.system}] if args.system else []
    messages = list(system)
    params = loaded.model.num_params() / 1e6
    out.write(f"Aupus | {loaded.source} | {params:.0f}M params | {device}\n{HELP}\n")
    if loaded.info.get("stage", "pretrain") == "pretrain":
        out.write("Note: this is a Base model (not fine-tuned for chat); expect it to ramble.\n")
    while True:
        try:
            line = read_line("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            out.write("\n")
            break
        if not line:
            continue
        if line in ("/exit", "/quit"):
            break
        if line == "/reset":
            messages = list(system)
            out.write("(conversation cleared)\n")
            continue
        if line.startswith("/"):
            out.write(f"unknown command {line.split()[0]}. {HELP}\n")
            continue
        messages.append({"role": "user", "content": line})
        out.write("aupus> ")
        try:
            reply, dropped = respond(loaded, messages, args.max_new_tokens, out, **sampling)
        except ValueError as e:
            messages.pop()
            out.write(f"({e})\n")
            continue
        out.write("\n")
        if dropped:
            out.write(f"(the {dropped} oldest messages no longer fit in the context window)\n")
        messages.append({"role": "assistant", "content": reply})
    return messages


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_model_args(parser)
    parser.add_argument("--system", default=None, help="optional system message")
    add_sampling_args(parser, temperature=0.7, max_new=256)
    args = parser.parse_args(argv)
    with contextlib.suppress(ImportError):  # readline: arrow keys and history in input() (not on Windows)
        import readline  # noqa: F401
    chat_loop(open_model(args), args)
