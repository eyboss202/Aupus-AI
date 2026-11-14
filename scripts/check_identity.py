"""Check ``data/identity.jsonl`` before training on it (see docs/IDENTITY.md).

Errors (the file can't be used; exit code 1):
  * a line that isn't valid JSON, or a conversation with the wrong roles or order;
  * a conversation longer than the context window (it would be cut).

Warnings (the file works, but probably isn't what you want):
  * fewer than 30 conversations, or the same user question twice;
  * something that looks personal: an email address, a phone number or a link
    (this file is public once committed);
  * text that doesn't look like English (Vulpes only learned English);
  * a reply that never says the name "Vulpes" in a conversation about its name;
  * very long replies (identity answers should be short).

Usage:
    python scripts/check_identity.py
    python scripts/check_identity.py --file data/identity.jsonl --context 1024
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from vulpes.data.sft import identity_conversations
from vulpes.tokenizer import Tokenizer
from vulpes.tokenizer.chat import render

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE = re.compile(r"\+?\d[\d ()-]{7,}\d")
LINK = re.compile(r"https?://|www\.", re.IGNORECASE)
NAME_QUESTION = re.compile(r"\bname\b|\bcall you\b|\bwho are (you|u)\b|\bwho r u\b", re.IGNORECASE)
MIN_CONVERSATIONS = 30
LONG_REPLY_WORDS = 80


def non_english_share(text: str) -> float:
    """Share of letters outside basic Latin: a cheap hint that a text isn't English."""
    letters = [c for c in text if c.isalpha()]
    return sum(1 for c in letters if ord(c) > 127) / len(letters) if letters else 0.0


def check(path: Path, tokenizer: Tokenizer, context: int) -> tuple[list[str], list[str], dict[str, float]]:
    """Return ``(errors, warnings, stats)`` for an identity file."""
    errors: list[str] = []
    warnings: list[str] = []
    # First the JSON shape of every line, so each mistake is reported with its line number.
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"line {n}: not valid JSON ({e.msg} at character {e.colno})")
            continue
        msgs = record.get("messages") if isinstance(record, dict) else None
        if not isinstance(msgs, list) or not all(
            isinstance(m, dict) and isinstance(m.get("role"), str) and isinstance(m.get("content"), str)
            for m in msgs
        ):
            errors.append(f'line {n}: needs {{"messages": [{{"role": ..., "content": ...}}, ...]}}')
    if errors:
        return errors, warnings, {}
    try:
        convs = identity_conversations(path)  # roles and turn order, with the line number
    except ValueError as e:
        return [str(e)], warnings, {}

    seen: Counter[str] = Counter()
    reply_tokens: list[int] = []
    for conv in convs:
        where = f"line {conv['id'].split('/')[1]}"
        msgs = conv["messages"]
        ids, _ = render(tokenizer, msgs)
        if len(ids) > context + 1:
            errors.append(f"{where}: {len(ids)} tokens, longer than the {context}-token context window")
        for m in msgs:
            text = m["content"]
            if not text.strip():
                errors.append(f"{where}: an empty {m['role']} message")
            for label, pattern in (("an email address", EMAIL), ("a phone number", PHONE), ("a link", LINK)):
                if pattern.search(text):
                    warnings.append(f"{where}: {m['role']} text contains {label} (this file is public)")
            if non_english_share(text) > 0.05:
                warnings.append(
                    f"{where}: {m['role']} text doesn't look like English (Vulpes is English-only)"
                )
            if m["role"] == "user":
                seen[text.strip().lower()] += 1
            if m["role"] == "assistant":
                reply_tokens.append(len(tokenizer.encode(text)))
                if len(text.split()) > LONG_REPLY_WORDS:
                    warnings.append(
                        f"{where}: a {len(text.split())}-word reply (identity answers work best short)"
                    )
        users = " ".join(m["content"] for m in msgs if m["role"] == "user")
        replies = " ".join(m["content"] for m in msgs if m["role"] == "assistant")
        if NAME_QUESTION.search(users) and "vulpes" not in replies.lower():
            warnings.append(f"{where}: asks about its name, but no reply says 'Vulpes'")
    for question, n in seen.items():
        if n > 1:
            warnings.append(f"the user question {question[:60]!r} appears {n} times (vary the wording)")
    if len(convs) < MIN_CONVERSATIONS:
        warnings.append(f"only {len(convs)} conversations; aim for 40-80 (docs/IDENTITY.md)")
    stats = {
        "conversations": len(convs),
        "multi_turn": sum(1 for c in convs if sum(m["role"] == "user" for m in c["messages"]) > 1),
        "replies": len(reply_tokens),
        "mean_reply_tokens": sum(reply_tokens) / len(reply_tokens) if reply_tokens else 0.0,
    }
    return errors, warnings, stats


def main() -> None:
    fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=fmt)
    parser.add_argument("--file", type=Path, default=Path("data/identity.jsonl"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer/vulpes-bpe-32k.json"))
    parser.add_argument("--context", type=int, default=1024)
    args = parser.parse_args()
    if not args.file.exists():
        raise SystemExit(f"{args.file} doesn't exist yet: see docs/IDENTITY.md for how to write it")

    errors, warnings, stats = check(args.file, Tokenizer.load(args.tokenizer), args.context)
    for e in errors:
        print(f"ERROR    {e}")
    for w in warnings:
        print(f"warning  {w}")
    if stats:
        print(
            f"\n{stats['conversations']} conversations ({stats['multi_turn']} multi-turn), "
            f"{stats['replies']} replies, {stats['mean_reply_tokens']:.0f} tokens per reply on average"
        )
    print("not usable yet: fix the errors above" if errors else "OK: ready for scripts/prepare_sft_data.py")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
