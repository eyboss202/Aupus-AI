"""Pre-tokenization: split text into chunks before BPE runs on each chunk.

Why pre-tokenize at all: if BPE ran on raw text it would learn merges that
cross word boundaries ("e t" in "the time"), wasting vocabulary on accidental
pairs and making tokens depend on neighbouring words. Splitting first means a
merge can never cross a chunk boundary, so tokens line up with words, numbers,
punctuation and whitespace.

The pattern follows GPT-2's, with two changes taken from later tokenizers
(GPT-4's cl100k, Llama 3):

* numbers are split into groups of at most 3 digits, so "2024" is "202" + "4"
  rather than an arbitrary memorised token. This keeps arithmetic and years
  consistent and stops the vocab filling up with numbers;
* newlines get their own chunks (``\\s*[\\r\\n]+``) and may trail punctuation,
  so code and paragraph breaks tokenize consistently.

GPT-2 uses the third-party ``regex`` module for ``\\p{L}`` (letter) and
``\\p{N}`` (number). We stay on the standard library ``re``, building the same
classes from what it has:

=================  ==============  ========================================
GPT-2 (``regex``)  here (``re``)   meaning
=================  ==============  ========================================
``\\p{L}``          ``[^\\W\\d_]``   word characters that are not digits or ``_``
``\\p{N}``          ``\\d``          Unicode decimal digits (category Nd)
``[^\\s\\p{L}\\p{N}]`` ``(?:[^\\s\\w]|_)`` everything else that is not whitespace
=================  ==============  ========================================

The one visible difference: non-decimal numerals such as "²" or "½" count as
letters here instead of numbers. They're rare in English text, so it doesn't
matter in practice.

**Losslessness.** Every character is whitespace, a decimal digit, ``_``, some
other ``\\w`` character or a non-``\\w`` non-space character, and each of those
classes is matched by at least one alternative. So the chunks always join back
into the original text. ``tests/test_tokenizer.py`` checks this on random
Unicode.
"""

from __future__ import annotations

import re

#: The pre-tokenization regex. Stored in every tokenizer file so a saved
#: tokenizer keeps working even if this default changes later.
PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"  # English contractions, kept separate: "don" + "'t"
    r"| ?[^\W\d_]+"  # a word, with its leading space: " hello"
    r"|\d{1,3}"  # numbers, at most 3 digits per chunk
    r"| ?(?:[^\s\w]|_)+[\r\n]*"  # punctuation / symbols, optional leading space
    r"|\s*[\r\n]+"  # newlines (with any whitespace before them)
    r"|\s+(?!\S)"  # trailing whitespace; leaves the last space for the next word
    r"|\s+"  # any other whitespace
)

_compiled: dict[str, re.Pattern[str]] = {}


def compile_pattern(pattern: str = PATTERN) -> re.Pattern[str]:
    """Compile (and cache) a pre-tokenization pattern."""
    if pattern not in _compiled:
        _compiled[pattern] = re.compile(pattern)
    return _compiled[pattern]


def pretokenize(text: str, pattern: str = PATTERN) -> list[str]:
    """Split ``text`` into chunks; ``"".join(pretokenize(t)) == t`` always holds."""
    return compile_pattern(pattern).findall(text)
