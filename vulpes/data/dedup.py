"""Exact and near-duplicate detection.

Why dedup: the web is full of copies (mirrors, syndicated articles, templated
pages). Training on the same text many times wastes compute and makes the
model memorise and regurgitate it. Two passes are used:

**Exact:** a 128-bit BLAKE2b hash of the normalised text. Identical documents
collide, so the first one is kept and the rest are dropped.

**Near-duplicates: MinHash + LSH** (Broder 1997; as used for GPT-3, The Pile
and FineWeb). Two documents are compared by the *Jaccard similarity* of their
sets of word 5-grams ("shingles"): |A ∩ B| / |A ∪ B|. Comparing every pair is
impossible, so:

1. *MinHash:* apply ``num_perm`` random hash functions to every shingle and
   keep each function's minimum. For any one function, P(min over A == min
   over B) is exactly the Jaccard similarity. So the fraction of equal
   entries in two signatures estimates it, using only ``num_perm`` numbers per
   document.
2. *LSH banding:* cut the signature into ``bands`` bands of ``rows`` values
   and hash each band. Similar documents very likely share at least one whole
   band; dissimilar ones almost never do (curve in ``DedupConfig``).

Documents are processed in a fixed order and the first of each near-duplicate
group is kept ("keep first"), so no all-pairs clustering is needed. Memory is
one Python int per band per kept document (~16 x 70 bytes ≈ 1.1 KB/doc), so
~3 GB for the full 2.7M-document run.

Hash functions are ``h(x) = (a*x + b) mod p`` with p = 2^32 - 5 (prime) and
32-bit CRC shingle hashes. Everything fits in uint64 without overflow, so
NumPy does all of it in C.
"""

from __future__ import annotations

import hashlib
import re
import zlib

import numpy as np

from vulpes.data.config import DedupConfig

_PRIME = np.uint64(4294967291)  # largest prime below 2^32
_WORDS = re.compile(r"\w+")
_CHUNK = 4096  # shingles per NumPy block (caps memory at num_perm * 4096 * 8 bytes)


def exact_hash(text: str) -> bytes:
    """128-bit content hash (also used for the train/val split)."""
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()


def shingles(text: str, n: int) -> np.ndarray:
    """Distinct word n-grams of the lowercased text, as 32-bit CRCs (in uint64)."""
    words = _WORDS.findall(text.lower())
    if len(words) <= n:
        grams = {" ".join(words)}
    else:
        grams = {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}
    return np.fromiter((zlib.crc32(g.encode("utf-8")) for g in grams), dtype=np.uint64, count=len(grams))


class MinHasher:
    """Computes MinHash signatures and LSH band keys for a fixed ``DedupConfig``."""

    def __init__(self, cfg: DedupConfig) -> None:
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        self._a = rng.integers(1, int(_PRIME), size=(cfg.num_perm, 1), dtype=np.uint64)
        self._b = rng.integers(0, int(_PRIME), size=(cfg.num_perm, 1), dtype=np.uint64)

    def signature(self, text: str) -> np.ndarray:
        """``num_perm`` minimum hash values (uint32)."""
        x = shingles(text, self.cfg.ngram)
        sig = np.full(self.cfg.num_perm, np.iinfo(np.uint64).max, dtype=np.uint64)
        for start in range(0, len(x), _CHUNK):
            block = (self._a * x[None, start : start + _CHUNK] + self._b) % _PRIME
            np.minimum(sig, block.min(axis=1), out=sig)
        return sig.astype(np.uint32)

    def band_keys(self, sig: np.ndarray) -> list[int]:
        """One 64-bit key per band; the band index is mixed in so bands never collide with each other."""
        rows = sig.reshape(self.cfg.bands, self.cfg.rows)
        return [
            int.from_bytes(hashlib.blake2b(band.tobytes(), digest_size=8, salt=i.to_bytes(16, "little")).digest())
            for i, band in enumerate(rows)
        ]


def estimated_jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    return float(np.mean(sig_a == sig_b))


class Deduplicator:
    """Stateful keep-first dedup over a stream of documents."""

    def __init__(self) -> None:
        self._exact: set[bytes] = set()
        self._bands: set[int] = set()

    def is_exact_duplicate(self, digest: bytes) -> bool:
        """True if seen before; otherwise remember it and return False."""
        if digest in self._exact:
            return True
        self._exact.add(digest)
        return False

    def is_near_duplicate(self, band_keys: list[int]) -> bool:
        """True if any band matches a kept document; otherwise remember this one."""
        if any(k in self._bands for k in band_keys):
            return True
        self._bands.update(band_keys)
        return False
