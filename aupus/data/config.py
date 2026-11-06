"""Data-pipeline settings (loaded from ``configs/data/*.yaml`` with the strict config loader)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FilterConfig:
    """Thresholds for the document quality filters (see ``filters.py``).

    Defaults follow the Gopher rules (Rae et al. 2021, appendix A), which
    FineWeb also uses, loosened where FineWeb-Edu has already been filtered
    upstream. They catch the junk that slips through: boilerplate, keyword
    lists, broken encodings and very short pages.
    """

    min_words: int = 50
    max_words: int = 100_000
    min_mean_word_length: float = 3.0
    max_mean_word_length: float = 10.0
    max_symbol_to_word_ratio: float = 0.1  # "#" and "…" per word
    max_ellipsis_line_fraction: float = 0.3  # lines ending in "..." or "…"
    max_bullet_line_fraction: float = 0.9  # lines starting with a bullet
    min_alpha_word_fraction: float = 0.8  # words containing at least one letter
    min_stop_words: int = 2  # distinct English stop words; a cheap "is this English prose" test
    max_duplicate_line_fraction: float = 0.3  # lines that repeat an earlier line
    max_replacement_char_fraction: float = 0.001  # U+FFFD: text that was decoded wrongly upstream


@dataclass(frozen=True)
class DedupConfig:
    """Near-duplicate detection with MinHash + LSH (see ``dedup.py``).

    With ``num_perm = bands * rows`` hashes, two documents become candidates
    when all ``rows`` values of at least one band agree. The probability of
    that for Jaccard similarity ``s`` is ``1 - (1 - s^rows)^bands``. For 16
    bands x 8 rows the 50% point is at s ~ 0.71: a pair with s = 0.8 is caught
    95% of the time, a pair with s = 0.5 only 6%.
    """

    num_perm: int = 128
    bands: int = 16
    ngram: int = 5  # shingle size in words
    seed: int = 1234

    def __post_init__(self) -> None:
        if self.num_perm % self.bands:
            raise ValueError(f"num_perm ({self.num_perm}) must be divisible by bands ({self.bands})")

    @property
    def rows(self) -> int:
        return self.num_perm // self.bands


@dataclass(frozen=True)
class DataConfig:
    """Everything that decides which tokens end up in the shards."""

    filters: FilterConfig = field(default_factory=FilterConfig)
    dedup: DedupConfig = field(default_factory=DedupConfig)
    val_fraction: float = 0.005  # 0.5% of documents, chosen by hash (stable across runs)
    shard_tokens: int = 100_000_000  # tokens per shard file (200 MB as uint16)
