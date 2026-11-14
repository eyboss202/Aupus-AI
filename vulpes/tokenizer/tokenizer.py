"""The Vulpes tokenizer: byte-level BPE encode / decode, special tokens, file format.

Id layout (for the 32,000-id Vulpes 0.1 vocab)::

    0 ..... 255     one token per byte value
    256 ... 31992   learned merges, in training order (merge i -> id 256 + i)
    31993 . 31999   special tokens (see special.py), in SPECIAL_TOKENS order

**Encoding** a chunk applies the learned merges by priority: of all adjacent
pairs in the chunk, merge the one learned *earliest*, and repeat until no pair
is in the merge table. This gives exactly the same tokens as replaying every
merge in training order. It's just much faster, because it only looks at pairs
that occur. English repeats the same words constantly, so each chunk's result
is cached.

**Decoding** concatenates each token's bytes and decodes the result as UTF-8.
A token can end in the middle of a multi-byte character (e.g. half of an emoji),
so decoding a *slice* of tokens may produce U+FFFD replacement characters.
Decoding a full ``encode`` output is always lossless.

**File format** (``*.json``, our own, human-readable)::

    {
      "format": "vulpes-bpe",
      "version": 1,
      "pattern": "<pre-tokenization regex>",
      "vocab_size": 32000,
      "special_tokens": {"<|bos|>": 31993, ...},
      "merges": [[left_id, right_id], ...]
    }

Merges are stored as id pairs (not strings) so the file is exact for byte
sequences that aren't valid UTF-8 on their own. ``dump_vocab`` writes a
readable list of every token for inspection.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from vulpes.tokenizer.pretokenize import PATTERN, compile_pattern
from vulpes.tokenizer.special import SPECIAL_TOKENS
from vulpes.tokenizer.train import Pair, count_chunks, merge_pair, train_bpe

FORMAT = "vulpes-bpe"
#: The name the format had before the project was renamed. Files written then still load (the
#: committed 32k tokenizer is one: rewriting it would change the SHA-256 that shards and checkpoints record).
LEGACY_FORMATS = ("aupus-bpe",)
VERSION = 1


class TokenizerError(ValueError):
    """Raised for malformed tokenizer files or invalid token ids."""


class Tokenizer:
    """Byte-level BPE tokenizer.

    Args:
        merges: learned merges in order; merge ``i`` creates id ``256 + i``.
        special_tokens: special token strings, given ids after the merges.
        pattern: pre-tokenization regex (see ``pretokenize.py``).
        cache_size: max number of cached chunk encodings (the cache is simply
            cleared when full; English chunk frequencies make this effective).
    """

    def __init__(
        self,
        merges: Sequence[Pair] = (),
        special_tokens: Sequence[str] = SPECIAL_TOKENS,
        pattern: str = PATTERN,
        cache_size: int = 500_000,
    ) -> None:
        self.pattern = pattern
        self._regex = compile_pattern(pattern)
        self.merges: list[Pair] = []
        self._vocab: list[bytes] = [bytes([i]) for i in range(256)]
        for i, merge in enumerate(merges):
            a, b = merge
            new_id = 256 + i
            if not (0 <= a < new_id and 0 <= b < new_id):
                raise TokenizerError(f"merge {i} {merge!r} refers to an id that doesn't exist yet")
            self.merges.append((a, b))
            self._vocab.append(self._vocab[a] + self._vocab[b])
        self._ranks: dict[Pair, int] = {pair: i for i, pair in enumerate(self.merges)}
        if len(self._ranks) != len(self.merges):
            raise TokenizerError("duplicate merges")

        if len(set(special_tokens)) != len(special_tokens) or any(not s for s in special_tokens):
            raise TokenizerError(f"special tokens must be unique and non-empty: {special_tokens!r}")
        first = len(self._vocab)
        self.special_tokens: dict[str, int] = {s: first + i for i, s in enumerate(special_tokens)}
        self._special_by_id: dict[int, str] = {i: s for s, i in self.special_tokens.items()}
        # Longest first, so a special token that is a prefix of another can't shadow it.
        alternatives = sorted(special_tokens, key=len, reverse=True)
        self._special_regex = re.compile("|".join(map(re.escape, alternatives))) if alternatives else None

        self._cache: dict[str, list[int]] = {}
        self._cache_size = cache_size

    # ------------------------------------------------------------------ basics

    @property
    def vocab_size(self) -> int:
        """Total number of ids: bytes + merges + special tokens."""
        return len(self._vocab) + len(self.special_tokens)

    def special_id(self, token: str) -> int:
        """Id of a special token, e.g. ``tok.special_id(EOS)``."""
        try:
            return self.special_tokens[token]
        except KeyError:
            raise TokenizerError(f"not a special token: {token!r}") from None

    def is_special(self, token_id: int) -> bool:
        return token_id in self._special_by_id

    def token_bytes(self, token_id: int) -> bytes:
        """The bytes a token stands for (special tokens: their UTF-8 name)."""
        if 0 <= token_id < len(self._vocab):
            return self._vocab[token_id]
        special = self._special_by_id.get(token_id)
        if special is None:
            raise TokenizerError(f"token id {token_id} out of range [0, {self.vocab_size})")
        return special.encode("utf-8")

    # ---------------------------------------------------------------- encoding

    def _encode_chunk(self, chunk: str) -> list[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk.encode("utf-8"))
        ranks = self._ranks
        while len(ids) >= 2:
            # The adjacent pair that was learned earliest (lowest rank) merges first.
            best = min(zip(ids, ids[1:], strict=False), key=lambda p: ranks.get(p, 1 << 62))
            rank = ranks.get(best)
            if rank is None:
                break
            ids = merge_pair(ids, best, 256 + rank)
        if len(self._cache) >= self._cache_size:
            self._cache.clear()
        self._cache[chunk] = ids
        return ids

    def encode_ordinary(self, text: str) -> list[int]:
        """Encode text, treating special-token strings as plain text."""
        out: list[int] = []
        for chunk in self._regex.findall(text):
            out.extend(self._encode_chunk(chunk))
        return out

    def encode(self, text: str, allow_special: bool = False) -> list[int]:
        """Encode text to token ids.

        Args:
            text: any Unicode string.
            allow_special: if True, occurrences of special-token strings (e.g.
                ``"<|eos|>"``) become their special ids. If False (default),
                they are encoded as ordinary text. Only enable it for text you
                built yourself, never for raw user input.
        """
        if not allow_special or self._special_regex is None:
            return self.encode_ordinary(text)
        out: list[int] = []
        start = 0
        for m in self._special_regex.finditer(text):
            out.extend(self.encode_ordinary(text[start : m.start()]))
            out.append(self.special_tokens[m.group()])
            start = m.end()
        out.extend(self.encode_ordinary(text[start:]))
        return out

    def encode_batch(self, texts: Iterable[str], allow_special: bool = False) -> list[list[int]]:
        return [self.encode(t, allow_special=allow_special) for t in texts]

    # ---------------------------------------------------------------- decoding

    def decode_bytes(self, ids: Iterable[int]) -> bytes:
        """Exact bytes for a sequence of ids (no UTF-8 decoding)."""
        return b"".join(self.token_bytes(i) for i in ids)

    def decode(self, ids: Iterable[int], skip_special: bool = False) -> str:
        """Decode ids to text. Partial UTF-8 sequences become U+FFFD."""
        if skip_special:
            ids = (i for i in ids if i not in self._special_by_id)
        return self.decode_bytes(ids).decode("utf-8", errors="replace")

    # ---------------------------------------------------------------- training

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int,
        special_tokens: Sequence[str] = SPECIAL_TOKENS,
        pattern: str = PATTERN,
        min_frequency: int = 2,
        num_workers: int = 1,
        progress: Callable[[int], None] | None = None,
    ) -> Tokenizer:
        """Train a tokenizer with exactly ``vocab_size`` ids (fewer if the corpus runs out of pairs)."""
        num_merges = vocab_size - 256 - len(special_tokens)
        if num_merges < 0:
            raise ValueError(
                f"vocab_size {vocab_size} is smaller than 256 bytes + {len(special_tokens)} specials"
            )
        counts = count_chunks(texts, pattern=pattern, num_workers=num_workers)
        merges = train_bpe(counts, num_merges, min_frequency=min_frequency, progress=progress)
        return cls(merges, special_tokens=special_tokens, pattern=pattern)

    # ------------------------------------------------------------- persistence

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "version": VERSION,
            "pattern": self.pattern,
            "vocab_size": self.vocab_size,
            "special_tokens": dict(self.special_tokens),
            "merges": [list(m) for m in self.merges],
        }

    def save(self, path: str | Path) -> None:
        """Write the tokenizer as JSON, one merge per line (diff-friendly)."""
        d = self.to_dict()
        merges = d.pop("merges")
        head = json.dumps(d, indent=2, ensure_ascii=False)[:-2]  # drop the closing "\n}"
        body = ",\n".join(f"    [{a}, {b}]" for a, b in merges)
        text = f'{head},\n  "merges": [\n{body}\n  ]\n}}\n' if merges else f'{head},\n  "merges": []\n}}\n'
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text, encoding="utf-8", newline="\n")

    @classmethod
    def from_dict(cls, d: Any) -> Tokenizer:
        if not isinstance(d, dict) or d.get("format") not in (FORMAT, *LEGACY_FORMATS):
            raise TokenizerError(f"not a {FORMAT} tokenizer file")
        if d.get("version") != VERSION:
            raise TokenizerError(f"unsupported tokenizer version {d.get('version')!r} (expected {VERSION})")
        expected_keys = {"format", "version", "pattern", "vocab_size", "special_tokens", "merges"}
        if set(d) != expected_keys:
            raise TokenizerError(f"tokenizer keys must be exactly {sorted(expected_keys)}, got {sorted(d)}")
        merges = d["merges"]
        if not isinstance(merges, list) or not all(
            isinstance(m, list) and len(m) == 2 and all(type(x) is int for x in m) for m in merges
        ):
            raise TokenizerError("merges must be a list of [int, int] pairs")
        specials = d["special_tokens"]
        if not isinstance(specials, dict) or not all(
            isinstance(k, str) and type(v) is int for k, v in specials.items()
        ):
            raise TokenizerError("special_tokens must map strings to ints")
        if not isinstance(d["pattern"], str):
            raise TokenizerError("pattern must be a string")
        ordered = sorted(specials, key=specials.__getitem__)
        tok = cls([tuple(m) for m in merges], special_tokens=ordered, pattern=d["pattern"])
        if tok.special_tokens != specials:
            raise TokenizerError("special token ids must directly follow the merges, in order")
        if tok.vocab_size != d["vocab_size"]:
            raise TokenizerError(
                f"vocab_size {d['vocab_size']} doesn't match the file contents ({tok.vocab_size})"
            )
        return tok

    @classmethod
    def load(cls, path: str | Path) -> Tokenizer:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        try:
            return cls.from_dict(data)
        except TokenizerError as e:
            raise TokenizerError(f"{path}: {e}") from None

    def dump_vocab(self, path: str | Path) -> None:
        """Write ``id <tab> token`` lines for humans (bytes shown with Python escapes)."""
        lines = []
        for i in range(self.vocab_size):
            if i in self._special_by_id:
                shown = self._special_by_id[i] + "  (special)"
            else:
                raw = self._vocab[i]
                try:
                    shown = repr(raw.decode("utf-8"))
                except UnicodeDecodeError:
                    shown = repr(raw)
            lines.append(f"{i}\t{shown}")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    def __repr__(self) -> str:
        return (
            f"Tokenizer(vocab_size={self.vocab_size}, merges={len(self.merges)}, "
            f"specials={len(self.special_tokens)})"
        )
