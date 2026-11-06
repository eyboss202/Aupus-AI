"""Document quality filters.

Each rule is a cheap heuristic for "this isn't prose a language model should
learn from". ``check`` returns the name of the first rule a document fails,
or ``None`` if it passes, so the pipeline can report *why* documents were
dropped (see the stats file each run writes).

The rules run after ``normalize``, so they see clean newlines and NFC text.
"""

from __future__ import annotations

import re

from aupus.data.config import FilterConfig

#: Gopher's stop-word list. Real English prose almost always uses at least two
#: of these; keyword lists, code dumps and other languages usually don't.
STOP_WORDS = frozenset({"the", "be", "to", "of", "and", "that", "have", "with"})
_BULLETS = ("•", "●", "○", "▪", "‣", "-", "*", "–", "—", "·")
_HAS_LETTER = re.compile(r"[^\W\d_]")
_BANNED = re.compile(r"lorem ipsum", re.IGNORECASE)


def check(text: str, cfg: FilterConfig) -> str | None:
    """Return the first failed rule's name, or None if ``text`` passes every rule."""
    if not text:
        return "empty"
    if "�" in text and text.count("�") / len(text) > cfg.max_replacement_char_fraction:
        return "replacement_chars"
    words = text.split()
    n = len(words)
    if n < cfg.min_words:
        return "too_short"
    if n > cfg.max_words:
        return "too_long"
    mean_len = sum(len(w) for w in words) / n
    if not cfg.min_mean_word_length <= mean_len <= cfg.max_mean_word_length:
        return "mean_word_length"
    if (text.count("#") + text.count("…")) / n > cfg.max_symbol_to_word_ratio:
        return "symbol_ratio"
    if sum(1 for w in words if _HAS_LETTER.search(w)) / n < cfg.min_alpha_word_fraction:
        return "alpha_words"
    lowered = {w.lower().strip(".,;:!?\"'()") for w in words}
    if len(STOP_WORDS & lowered) < cfg.min_stop_words:
        return "stop_words"

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if lines:
        if sum(1 for line in lines if line.endswith(("...", "…"))) / len(lines) > cfg.max_ellipsis_line_fraction:
            return "ellipsis_lines"
        if sum(1 for line in lines if line.startswith(_BULLETS)) / len(lines) > cfg.max_bullet_line_fraction:
            return "bullet_lines"
        if (len(lines) - len(set(lines))) / len(lines) > cfg.max_duplicate_line_fraction:
            return "duplicate_lines"
    if _BANNED.search(text):
        return "banned_phrase"
    return None
