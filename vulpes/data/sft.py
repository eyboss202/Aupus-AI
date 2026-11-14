"""Supervised fine-tuning data: human-written conversations -> masked, padded training batches.

**Sources** (docs/DATA.md has the full table and licenses). All of it is written by people:

* **OpenAssistant oasst2** (Apache-2.0): conversation *trees*. Volunteers wrote
  prompts and replies, and other volunteers ranked sibling replies. We keep the
  English trees and follow the best-ranked reply at every level ("top-1 branch"),
  so each tree gives one conversation. oasst2 contains all of oasst1 (only 8 of
  oasst1's 88,838 messages are missing from it), so oasst1 isn't downloaded separately;
* **Databricks Dolly 15k** (CC BY-SA 3.0): single-turn instruction/response pairs
  written by Databricks employees, some with a reference passage (``context``);
* **identity** (``data/identity.jsonl``, ours): a few dozen hand-written turns about
  what Vulpes is, so the model answers questions about itself truthfully.

Conversations in which the *assistant* calls itself Open Assistant (or mentions
LAION, the organisation behind it) are dropped, so Vulpes doesn't learn someone
else's identity. All text goes through the same ``normalize`` as pretraining.

**On disk** (``data/sft/{train,val}.jsonl``), one conversation per line::

    {"id": "oasst/<tree id>", "source": "oasst2", "messages": [{"role": "user", "content": ...}, ...]}

The val split is by hash of the conversation id (as in pretraining), so it never
depends on order.

**Batches.** Each conversation is rendered with the chat template
(``vulpes/tokenizer/chat.py``); targets outside assistant replies are set to
``IGNORE_INDEX``. Conversations vary from ~20 to 1,000+ tokens, so batches are
padded to their longest member. To waste less compute on padding, examples are
shuffled, cut into groups of ``50 * batch_size``, sorted by length within a group,
and batched; then the batch order is shuffled. Batches are of similar lengths,
but the data order is still random. Like ``PretrainLoader``, the whole state is
``(epoch, position)``.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vulpes.data.dedup import exact_hash
from vulpes.data.normalize import normalize
from vulpes.data.shards import is_val
from vulpes.model.transformer import IGNORE_INDEX
from vulpes.tokenizer import PAD, Tokenizer
from vulpes.tokenizer.chat import render, validate

Conversation = dict[str, Any]  # {"id", "source", "messages"}

#: An assistant reply that names another assistant's identity (case-insensitive).
FOREIGN_IDENTITY = re.compile(r"open[ -]?assistant|\blaion\b", re.IGNORECASE)
#: Wikipedia-style citation marks ("[12]") that Dolly copied along with its passages.
_CITATION = re.compile(r"\[\d{1,3}\]")


# ---------------------------------------------------------------- sources


def oasst_conversations(
    rows: Iterable[dict[str, Any]], lang: str = "en", stats: Counter | None = None
) -> Iterator[Conversation]:
    """Top-1 branch of every usable ``lang`` tree in an OpenAssistant messages table.

    A message is usable if it is in ``lang``, not deleted and passed review
    (``review_result`` is True). From the root prompt, each step takes the
    usable child with the best (lowest) rank. Prompter replies aren't ranked, so
    among those the earliest wins. A conversation must end with an assistant reply.
    """
    stats = stats if stats is not None else Counter()
    messages = {r["message_id"]: r for r in rows}
    children: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in messages.values():
        if r["parent_id"] is not None:
            children[r["parent_id"]].append(r)

    def usable(r: dict[str, Any]) -> bool:
        return r["lang"] == lang and not r["deleted"] and r["review_result"] is True and bool(r["text"].strip())

    def best_child(r: dict[str, Any]) -> dict[str, Any] | None:
        options = [c for c in children[r["message_id"]] if usable(c)]
        rank = lambda c: (math.inf if c["rank"] is None else c["rank"], c["created_date"])  # noqa: E731
        return min(options, key=rank) if options else None

    roots = sorted((r for r in messages.values() if r["parent_id"] is None), key=lambda r: r["message_tree_id"])
    for root in roots:
        if not usable(root) or root["role"] != "prompter":
            continue
        stats["trees"] += 1
        chain = [root]
        while (nxt := best_child(chain[-1])) is not None:
            chain.append(nxt)
        if chain[-1]["role"] == "prompter":
            chain.pop()  # a question nobody (usably) answered
        if len(chain) < 2:
            stats["dropped_no_reply"] += 1
            continue
        turns = [
            {"role": "user" if m["role"] == "prompter" else "assistant", "content": normalize(m["text"])}
            for m in chain
        ]
        if any(t["role"] == "assistant" and FOREIGN_IDENTITY.search(t["content"]) for t in turns):
            stats["dropped_foreign_identity"] += 1
            continue
        stats["kept"] += 1
        yield {"id": f"oasst/{root['message_tree_id']}", "source": "oasst2", "messages": turns}


def dolly_conversations(records: Iterable[dict[str, Any]], stats: Counter | None = None) -> Iterator[Conversation]:
    """One single-turn conversation per Dolly record: instruction (+ passage) -> response."""
    stats = stats if stats is not None else Counter()
    for i, r in enumerate(records):
        instruction = normalize(r["instruction"])
        context = normalize(_CITATION.sub("", r.get("context") or ""))
        response = normalize(_CITATION.sub("", r["response"]))
        if not instruction or not response:
            stats["dropped_empty"] += 1
            continue
        user = f"{instruction}\n\n{context}" if context else instruction
        stats["kept"] += 1
        yield {
            "id": f"dolly/{i}",
            "source": "dolly",
            "messages": [{"role": "user", "content": user}, {"role": "assistant", "content": response}],
        }


def identity_conversations(path: str | Path) -> list[Conversation]:
    """Read the hand-written identity set: one ``{"messages": [...]}`` object per line."""
    out = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            messages = [{"role": m["role"], "content": normalize(m["content"])} for m in record["messages"]]
            try:
                validate(messages)
            except ValueError as e:
                raise ValueError(f"{path}:{n}: {e}") from None
            out.append({"id": f"identity/{n}", "source": "identity", "messages": messages})
    return out


def split(conversations: Iterable[Conversation], val_fraction: float) -> tuple[list[Conversation], list[Conversation]]:
    """Deterministic train/val split by hash of the conversation id."""
    train, val = [], []
    for c in conversations:
        (val if is_val(exact_hash(c["id"]), val_fraction) else train).append(c)
    return train, val


def write_jsonl(conversations: Iterable[Conversation], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for c in conversations:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> list[Conversation]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------- tokenized examples


@dataclass
class SFTDataset:
    """Tokenized conversations: ``inputs[i]`` and ``targets[i]`` have the same length."""

    inputs: list[np.ndarray] = field(default_factory=list)
    targets: list[np.ndarray] = field(default_factory=list)  # IGNORE_INDEX where nothing is learned
    truncated: int = 0  # conversations cut at the context length
    dropped: int = 0  # conversations with nothing left to learn after truncation

    def __len__(self) -> int:
        return len(self.inputs)

    @property
    def lengths(self) -> np.ndarray:
        return np.array([len(x) for x in self.inputs])

    @property
    def target_tokens(self) -> int:
        return int(sum((t != IGNORE_INDEX).sum() for t in self.targets))


def tokenize(tokenizer: Tokenizer, conversations: Sequence[Conversation], seq_len: int) -> SFTDataset:
    """Render, shift and mask every conversation; cut those longer than ``seq_len`` inputs.

    A conversation that doesn't fit keeps its first ``seq_len + 1`` tokens: the
    model still learns from the replies (or part of a reply) that fit. One whose
    first reply starts beyond the cut has nothing to learn and is dropped.
    """
    ds = SFTDataset()
    for conv in conversations:
        ids, mask = render(tokenizer, conv["messages"])
        if len(ids) > seq_len + 1:
            ids, mask = ids[: seq_len + 1], mask[: seq_len + 1]
            ds.truncated += 1
        x = np.array(ids[:-1], dtype=np.int64)
        y = np.where(np.array(mask[1:]), np.array(ids[1:], dtype=np.int64), IGNORE_INDEX)
        if not (y != IGNORE_INDEX).any():
            ds.dropped += 1
            continue
        ds.inputs.append(x)
        ds.targets.append(y)
    return ds


def collate(ds: SFTDataset, idx: Sequence[int], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad examples ``idx`` into ``(inputs, targets)`` int64 tensors of equal width."""
    width = max(len(ds.inputs[i]) for i in idx)
    x = np.full((len(idx), width), pad_id, dtype=np.int64)
    y = np.full((len(idx), width), IGNORE_INDEX, dtype=np.int64)
    for row, i in enumerate(idx):
        n = len(ds.inputs[i])
        x[row, :n] = ds.inputs[i]
        y[row, :n] = ds.targets[i]
    return torch.from_numpy(x), torch.from_numpy(y)


class SFTLoader:
    """Infinite iterator of padded ``(inputs, targets)`` batches, length-grouped and resumable.

    Args:
        dataset: tokenized conversations.
        batch_size: conversations per batch (the micro-batch).
        pad_id: id used for padding inputs (its targets are always ignored).
        seed: base seed; epoch ``e`` uses ``default_rng([seed, e])``.
        group: batches per sorting group (see the module docstring).
    """

    def __init__(self, dataset: SFTDataset, batch_size: int, pad_id: int, seed: int = 0, group: int = 50) -> None:
        if len(dataset) == 0:
            raise ValueError("empty SFT dataset")
        self.dataset, self.batch_size, self.pad_id = dataset, batch_size, pad_id
        self.seed, self.group = seed, group
        self.epoch, self.position = 0, 0
        self._batches = self._epoch_batches(0)

    @property
    def batches_per_epoch(self) -> int:
        return math.ceil(len(self.dataset) / self.batch_size)

    def _epoch_batches(self, epoch: int) -> list[np.ndarray]:
        rng = np.random.default_rng([self.seed, epoch])
        order = rng.permutation(len(self.dataset))
        lengths = self.dataset.lengths
        batches = []
        span = self.batch_size * self.group
        for start in range(0, len(order), span):
            chunk = order[start : start + span]
            chunk = chunk[np.argsort(lengths[chunk], kind="stable")]
            batches += [chunk[i : i + self.batch_size] for i in range(0, len(chunk), self.batch_size)]
        return [batches[i] for i in rng.permutation(len(batches))]

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.position == len(self._batches):
            self.epoch += 1
            self.position = 0
            self._batches = self._epoch_batches(self.epoch)
        idx = self._batches[self.position]
        self.position += 1
        return collate(self.dataset, idx, self.pad_id)

    def state_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "position": self.position,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "num_examples": len(self.dataset),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for key in ("seed", "batch_size", "num_examples"):
            mine = self.state_dict()[key]
            if state[key] != mine:
                raise ValueError(f"loader state mismatch for {key}: checkpoint {state[key]}, current {mine}")
        self.epoch, self.position = state["epoch"], state["position"]
        self._batches = self._epoch_batches(self.epoch)


def sft_eval_batches(
    dataset: SFTDataset, batch_size: int, pad_id: int
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """One pass over every example, sorted by length (fixed order: comparable across evals)."""
    order = np.argsort(dataset.lengths, kind="stable")
    for start in range(0, len(order), batch_size):
        yield collate(dataset, order[start : start + batch_size], pad_id)


def pad_id(tokenizer: Tokenizer) -> int:
    return tokenizer.special_id(PAD)
