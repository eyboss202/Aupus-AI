"""Zero-shot multiple-choice benchmarks: HellaSwag, ARC-Easy and PIQA.

All three are scored the same way. A model never "answers" a question: for
each candidate answer we compute how likely the model thinks that text is as a
continuation of the question, and the most likely candidate is its pick. So a
base model that has never seen the question format can still be measured,
and no answer has to be parsed out of generated text.

Each raw dataset record is turned into an ``Example``: a context string, the
candidate continuations (each starting with a space, so it tokenizes as it
would in running text), and the index of the right one. The prompt formats
follow EleutherAI's lm-evaluation-harness, so our numbers can be compared with
published results for GPT-2, Pythia, SmolLM and others:

=============  ====================================================  ========================
task           context                                               continuation
=============  ====================================================  ========================
hellaswag      "<activity>: <first sentence> <start of 2nd>"          " <ending>"  (4 choices)
arc_easy       "Question: <question>\\nAnswer:"                       " <answer>"  (3-5 choices)
piqa           "Question: <goal>\\nAnswer:"                           " <solution>" (2 choices)
=============  ====================================================  ========================

Which splits: HellaSwag validation (10,042), ARC-Easy test (2,376), PIQA validation
(1,838). PIQA's and HellaSwag's test labels are not public; everyone reports these.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

#: Chance accuracy, for reading results: HellaSwag has 4 choices, PIQA 2, ARC-Easy mostly 4.
RANDOM_BASELINE = {"hellaswag": 0.25, "arc_easy": 0.25, "piqa": 0.5}


@dataclass(frozen=True)
class Example:
    """One multiple-choice question in a task-independent form."""

    id: str
    context: str
    choices: list[str]  # continuations, each starting with a space
    label: int  # index of the correct continuation

    def __post_init__(self) -> None:
        if not 0 <= self.label < len(self.choices):
            raise ValueError(f"{self.id}: label {self.label} out of range for {len(self.choices)} choices")


def _maybe_literal(value: Any) -> Any:
    """Some parquet exports store lists/dicts as their Python repr; accept both forms."""
    if isinstance(value, str) and value[:1] in "[{":
        return ast.literal_eval(value)
    return value


def _text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _hellaswag_clean(text: str) -> str:
    """WikiHow contexts contain markup like ``[title]`` and ``[step]``; drop it (as lm-eval does)."""
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def hellaswag(record: dict[str, Any]) -> Example:
    ctx = record["ctx_a"] + " " + record["ctx_b"].capitalize()
    endings = _maybe_literal(record["endings"])
    context = _hellaswag_clean(record["activity_label"] + ": " + ctx)
    return Example(
        # `ind` alone repeats within the validation split, so add a hash of the text.
        id=f"hellaswag/{record['ind']}-{_text_hash(context + endings[0])}",
        context=context,
        choices=[" " + _hellaswag_clean(e) for e in endings],
        label=int(record["label"]),
    )


def arc(record: dict[str, Any]) -> Example:
    choices = _maybe_literal(record["choices"])
    return Example(
        id=f"arc_easy/{record['id']}",
        context=f"Question: {record['question']}\nAnswer:",
        choices=[" " + t for t in choices["text"]],
        label=list(choices["label"]).index(record["answerKey"]),  # keys are "A".."E" or "1".."5"
    )


def piqa(record: dict[str, Any]) -> Example:
    return Example(
        id="piqa/" + _text_hash(record["goal"] + record["sol1"]),  # PIQA has no ids of its own
        context=f"Question: {record['goal']}\nAnswer:",
        choices=[" " + record["sol1"], " " + record["sol2"]],
        label=int(record["label"]),
    )


#: Raw-record converters, by task name.
CONVERTERS: dict[str, Callable[[dict[str, Any]], Example]] = {
    "hellaswag": hellaswag,
    "arc_easy": arc,
    "piqa": piqa,
}
TASKS = tuple(CONVERTERS)


def write_examples(examples: Iterable[Example], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(asdict(ex), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_examples(path: str | Path) -> Iterator[Example]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield Example(**json.loads(line))
