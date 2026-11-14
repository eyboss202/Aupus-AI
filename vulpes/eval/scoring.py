"""Log-likelihood scoring of continuations, and multiple-choice accuracy.

For a context ``c`` and a candidate continuation ``a = a_1 .. a_n`` (token ids),
the model's score is

    log P(a | c) = sum_i log P(a_i | c, a_1 .. a_{i-1})

which one forward pass over ``c + a`` gives us: the logits at the position
before each ``a_i`` say how likely ``a_i`` is. Two accuracies are reported:

* ``acc``: pick the candidate with the highest ``log P(a | c)``. Longer answers
  have more tokens, so more chances to lose probability, which biases this towards short answers;
* ``acc_norm``: divide by the candidate's length in characters first (lm-eval's
  definition). That removes most of the length bias, and is the standard headline
  number for HellaSwag.

Batching: candidates are sorted by length so a batch pads as little as possible,
and padding goes on the right. With causal attention a token never sees later
positions, so right padding can't change any score (``tests/test_eval.py`` checks
that batched and one-at-a-time scores agree).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from vulpes.eval.tasks import Example
from vulpes.model import Transformer
from vulpes.tokenizer import EOS, Tokenizer


@dataclass(frozen=True)
class Score:
    logprob: float  # log P(continuation | context), natural log
    greedy: bool  # True if greedy decoding would have produced exactly this continuation


@torch.no_grad()
def score_continuations(
    model: Transformer,
    requests: Sequence[tuple[Sequence[int], Sequence[int]]],
    batch_size: int = 16,
    progress: bool = False,
) -> list[Score]:
    """Score ``(context_ids, continuation_ids)`` pairs. Results come back in input order.

    A pair longer than the context window keeps its continuation and loses the
    *start* of its context (the end of the context matters most for the next token).
    """
    window = model.cfg.context_length
    device = next(model.parameters()).device
    prepared = []
    for ctx, cont in requests:
        if not cont:
            raise ValueError("every continuation needs at least one token")
        seq = [*ctx, *cont][-(window + 1) :]
        prepared.append((seq, min(len(cont), len(seq) - 1)))

    order = sorted(range(len(prepared)), key=lambda i: -len(prepared[i][0]))
    batches = range(0, len(order), batch_size)
    if progress:
        from tqdm import tqdm

        batches = tqdm(batches, unit="batch", leave=False)
    results: list[Score | None] = [None] * len(prepared)
    was_training = model.training
    model.eval()
    try:
        for start in batches:
            idx = order[start : start + batch_size]
            width = max(len(prepared[i][0]) for i in idx) - 1
            x = torch.zeros(len(idx), width, dtype=torch.long)
            for row, i in enumerate(idx):
                seq = prepared[i][0]
                x[row, : len(seq) - 1] = torch.tensor(seq[:-1])
            logits = model(x.to(device)).logits
            for row, i in enumerate(idx):
                seq, n = prepared[i]
                end = len(seq) - 1  # logits[end - 1] predicts seq[end], the last token
                logp = F.log_softmax(logits[row, end - n : end].float(), dim=-1)  # (n, vocab)
                target = torch.tensor(seq[-n:], device=device)
                results[i] = Score(
                    logprob=float(logp.gather(1, target[:, None]).sum()),
                    greedy=bool((logp.argmax(-1) == target).all()),
                )
    finally:
        model.train(was_training)
    return results  # type: ignore[return-value]  # every slot was filled


def encode_pair(tokenizer: Tokenizer, context: str, continuation: str) -> tuple[list[int], list[int]]:
    """Token ids of context and continuation, split the way they'd be in running text.

    The whole string is encoded once and split after the context's tokens. Every
    continuation starts with a space, which always begins a new pre-tokenizer chunk,
    so the split is clean. The fallback only exists for unusual input. The context
    starts with ``<|eos|>``: in pretraining a new document begins right after it.
    """
    ctx = tokenizer.encode(context)
    whole = tokenizer.encode(context + continuation)
    cont = whole[len(ctx) :] if whole[: len(ctx)] == ctx else tokenizer.encode(continuation)
    return [tokenizer.special_id(EOS), *ctx], cont


@dataclass(frozen=True)
class TaskResult:
    task: str
    n: int
    acc: float
    acc_norm: float

    @property
    def acc_stderr(self) -> float:
        """Standard error of ``acc`` (binomial): differences smaller than ~2x this are noise."""
        return math.sqrt(self.acc * (1 - self.acc) / max(self.n, 1))


def run_task(
    task: str,
    model: Transformer,
    tokenizer: Tokenizer,
    examples: Iterable[Example],
    batch_size: int = 16,
    progress: bool = False,
) -> TaskResult:
    """Zero-shot accuracy of ``model`` on ``examples``."""
    examples = list(examples)
    requests = [encode_pair(tokenizer, ex.context, choice) for ex in examples for choice in ex.choices]
    scores = iter(score_continuations(model, requests, batch_size, progress))
    correct = correct_norm = 0
    for ex in examples:
        raw = [next(scores).logprob for _ in ex.choices]
        norm = [lp / max(len(choice), 1) for lp, choice in zip(raw, ex.choices, strict=True)]
        correct += raw.index(max(raw)) == ex.label
        correct_norm += norm.index(max(norm)) == ex.label
    n = len(examples)
    return TaskResult(task, n, correct / max(n, 1), correct_norm / max(n, 1))
