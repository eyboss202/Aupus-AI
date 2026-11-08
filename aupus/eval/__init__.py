"""Evaluation: validation perplexity and zero-shot log-likelihood benchmarks (Milestone 9)."""

from aupus.eval.perplexity import evaluate_loss
from aupus.eval.scoring import Score, TaskResult, encode_pair, run_task, score_continuations
from aupus.eval.tasks import CONVERTERS, RANDOM_BASELINE, TASKS, Example, read_examples, write_examples

__all__ = [
    "CONVERTERS",
    "RANDOM_BASELINE",
    "TASKS",
    "Example",
    "Score",
    "TaskResult",
    "encode_pair",
    "evaluate_loss",
    "read_examples",
    "run_task",
    "score_continuations",
    "write_examples",
]
