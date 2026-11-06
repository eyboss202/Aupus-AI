"""Text normalisation: make equivalent text identical before filtering, dedup and tokenizing.

The same visible text can be stored in different ways on the web:

* Unicode has two ways to write "é": one code point (U+00E9), or "e" plus a
  combining accent (U+0065 U+0301). **NFC** picks the composed form, so the
  tokenizer, the dedup hashes and the filters all see one spelling;
* Windows (``\\r\\n``), old Mac (``\\r``) and Unicode line/paragraph separators
  all become ``\\n``;
* invisible characters that only break words apart (zero-width space, soft
  hyphen, byte-order mark) and C0 control characters are removed. Tab and
  newline are kept. The zero-width *joiner* is kept too, because emoji
  sequences need it;
* trailing spaces on lines are dropped, and 3+ newlines collapse to one blank line.

This is deliberately light. It never rewrites words, quotes or punctuation,
because the model should learn real text. ``normalize`` is idempotent: running
it twice changes nothing (fuzz-tested).
"""

from __future__ import annotations

import re
import unicodedata

_NEWLINES = re.compile(r"\r\n?|[  \x85]")
# C0 controls except \t (09) and \n (0a), DEL, soft hyphen, zero-width space/non-joiner,
# word joiner and BOM. (\r was already turned into \n.)
_INVISIBLE = re.compile("[\x00-\x08\x0b-\x1f\x7f­​‌⁠﻿]")
_TRAILING_SPACE = re.compile(r"[ \t]+\n")
_MANY_NEWLINES = re.compile(r"\n{3,}")


def normalize(text: str) -> str:
    """Return the normalised form of one document (see module docstring)."""
    text = _NEWLINES.sub("\n", text)
    text = _INVISIBLE.sub("", text)
    text = _TRAILING_SPACE.sub("\n", text)
    text = _MANY_NEWLINES.sub("\n\n", text)
    # NFC goes last: removing an invisible character can put a letter next to a
    # combining accent ("a" + soft hyphen + U+0301), which only then composes.
    return unicodedata.normalize("NFC", text).strip()
