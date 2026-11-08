"""The Aupus chat template: how a conversation becomes token ids, and which ids are trained on.

A conversation is a list of messages ``{"role": ..., "content": ...}``: an
optional ``system`` message first, then ``user`` and ``assistant`` turns
alternating, starting with ``user``. It is laid out as::

    <|eos|> [<|system|> system text <|end|>]
            <|user|> question <|end|> <|assistant|> reply <|end|>
            <|user|> ...      <|end|> <|assistant|> ...   <|end|>

* ``<|eos|>`` first: in pretraining, a new document always starts right after
  ``<|eos|>``, so this tells the model "something new begins here" in a way it
  already understands (the same choice as for training samples, docs/PLAN.md decision 25);
* each turn is a role token, the text, and ``<|end|>``. Special tokens can't
  appear in text by accident: message content is always encoded with
  ``allow_special=False``, so a user typing ``<|assistant|>`` just produces the
  characters ``<``, ``|``, ``assistant``... (decision 9);
* no newlines or spaces around the special tokens: they are unambiguous on their own.

**Loss mask (SFT).** Only the assistant's reply tokens *and* the ``<|end|>``
closing each reply are trained on. Training on ``<|end|>`` is what teaches the
model to stop. The user's words, the system prompt and the role tokens are
context only: the model should learn to answer, not to imitate users.

**Generation.** ``prompt_ids`` renders a conversation that ends with a user turn
and appends ``<|assistant|>``. The model writes the reply and we stop at ``<|end|>``.
"""

from __future__ import annotations

from collections.abc import Sequence

from aupus.tokenizer.special import ASSISTANT, END, EOS, SYSTEM, USER
from aupus.tokenizer.tokenizer import Tokenizer

Message = dict[str, str]
ROLE_TOKENS = {"system": SYSTEM, "user": USER, "assistant": ASSISTANT}


class ChatFormatError(ValueError):
    """The messages don't form a valid conversation."""


def validate(messages: Sequence[Message], *, for_generation: bool = False) -> None:
    """Check roles and order: [system], user, assistant, user, assistant, ...

    A training conversation must end with an assistant reply (there is something
    to learn); a generation prompt must end with a user turn (there is something to answer).
    """
    if not messages:
        raise ChatFormatError("empty conversation")
    turns = list(messages)
    for m in turns:
        if set(m) != {"role", "content"}:
            raise ChatFormatError(f"a message needs exactly 'role' and 'content', got {sorted(m)}")
        if m["role"] not in ROLE_TOKENS:
            raise ChatFormatError(f"unknown role {m['role']!r} (use {', '.join(ROLE_TOKENS)})")
        if not isinstance(m["content"], str):
            raise ChatFormatError("message content must be a string")
    if turns[0]["role"] == "system":
        turns = turns[1:]
    for i, m in enumerate(turns):
        expected = "user" if i % 2 == 0 else "assistant"
        if m["role"] != expected:
            raise ChatFormatError(f"turn {i} should be {expected!r}, got {m['role']!r}")
    last = turns[-1]["role"] if turns else "system"
    if for_generation and last != "user":
        raise ChatFormatError("a generation prompt must end with a user turn")
    if not for_generation and last != "assistant":
        raise ChatFormatError("a training conversation must end with an assistant turn")


def render(tokenizer: Tokenizer, messages: Sequence[Message]) -> tuple[list[int], list[bool]]:
    """Token ids of a whole conversation, and for each id whether it is a training target.

    ``mask[i]`` is True when token ``i`` should be *predicted* (an assistant reply
    token or the ``<|end|>`` after it). For training, inputs are ``ids[:-1]`` and
    targets ``ids[1:]``, with targets ignored wherever ``mask[1:]`` is False.
    """
    validate(messages)
    end = tokenizer.special_id(END)
    ids, mask = [tokenizer.special_id(EOS)], [False]
    for m in messages:
        body = tokenizer.encode(m["content"])  # never allow_special: content is data
        learn = m["role"] == "assistant"
        ids += [tokenizer.special_id(ROLE_TOKENS[m["role"]]), *body, end]
        mask += [False, *([learn] * len(body)), learn]
    return ids, mask


def prompt_ids(tokenizer: Tokenizer, messages: Sequence[Message]) -> list[int]:
    """Ids for generating the next assistant reply: the conversation so far + ``<|assistant|>``."""
    validate(messages, for_generation=True)
    end = tokenizer.special_id(END)
    ids = [tokenizer.special_id(EOS)]
    for m in messages:
        ids += [tokenizer.special_id(ROLE_TOKENS[m["role"]]), *tokenizer.encode(m["content"]), end]
    return [*ids, tokenizer.special_id(ASSISTANT)]
