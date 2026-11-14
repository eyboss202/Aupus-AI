"""Training a byte-level BPE vocabulary.

Byte-level BPE in one paragraph: start with the 256 possible byte values as
tokens, so any text (any language, emoji, binary junk) can be encoded and
nothing is ever "unknown". Then repeat: find the most frequent pair of adjacent
tokens in the corpus, add it to the vocabulary as a new token, and replace
every occurrence. After N merges the vocabulary has 256 + N tokens, and common
words and word-pieces are single tokens.

Training is done in two steps:

1. ``count_chunks``: pre-tokenize the corpus (see ``pretokenize.py``) and count
   how often each distinct chunk occurs. English has a heavy-tailed word
   distribution, so a few hundred MB of text gives only a few million distinct
   chunks. Everything after this step works on (chunk, count) pairs instead of
   the raw text, which is what makes pure-Python training feasible.

2. ``train_bpe``: the merge loop. The obvious version re-counts every pair
   after every merge, which costs O(merges x corpus). We keep **incremental
   pair counts** instead:

   * ``pair_counts[p]``: how often pair ``p`` occurs, weighted by chunk count;
   * ``where[p]``: the ids of the chunks that contain ``p``;
   * a max-heap of ``(count, pair)`` to find the best pair quickly.

   A merge only touches the chunks listed in ``where[best]`` and only changes
   counts of pairs inside those chunks. Heap entries become stale when a count
   changes. Instead of updating them in place (which ``heapq`` can't do), we
   push a fresh entry and skip popped entries whose count no longer matches
   ("lazy deletion").

   Ties are broken by the smallest ``(left_id, right_id)`` pair, so training is
   deterministic. ``tests/test_tokenizer.py`` checks this loop against a naive
   re-count-everything reference trainer.
"""

from __future__ import annotations

import heapq
import multiprocessing
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping

from vulpes.tokenizer.pretokenize import PATTERN, compile_pattern

Pair = tuple[int, int]


def merge_pair(ids: list[int], pair: Pair, new_id: int) -> list[int]:
    """Replace every occurrence of ``pair`` in ``ids`` with ``new_id``, left to right.

    Occurrences are non-overlapping and found left to right, so ``a a a`` with
    pair ``(a, a)`` becomes ``new a``. Training and encoding must agree on this
    rule, which is why both use this function.
    """
    a, b = pair
    out: list[int] = []
    i, n = 0, len(ids)
    while i < n:
        if i + 1 < n and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def _count_batch(args: tuple[list[str], str]) -> Counter[str]:
    texts, pattern = args
    regex = compile_pattern(pattern)
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(regex.findall(text))
    return counts


def _batches(texts: Iterable[str], size: int, pattern: str) -> Iterator[tuple[list[str], str]]:
    batch: list[str] = []
    for text in texts:
        batch.append(text)
        if len(batch) == size:
            yield batch, pattern
            batch = []
    if batch:
        yield batch, pattern


def count_chunks(
    texts: Iterable[str],
    pattern: str = PATTERN,
    num_workers: int = 1,
    batch_size: int = 512,
) -> Counter[str]:
    """Count distinct pre-tokenized chunks over an iterable of documents.

    With ``num_workers > 1`` batches of documents are pre-tokenized in worker
    processes (the regex is the slow part and Python threads would hold the
    GIL). Counting is order-independent, so the result is identical either way.
    """
    total: Counter[str] = Counter()
    batches = _batches(texts, batch_size, pattern)
    if num_workers <= 1:
        for batch in batches:
            total.update(_count_batch(batch))
        return total
    # "spawn" behaves the same on Linux, macOS and Windows (fork isn't available on Windows).
    with multiprocessing.get_context("spawn").Pool(num_workers) as pool:
        for counts in pool.imap_unordered(_count_batch, batches, chunksize=1):
            total.update(counts)
    return total


def train_bpe(
    chunk_counts: Mapping[str, int],
    num_merges: int,
    min_frequency: int = 2,
    progress: Callable[[int], None] | None = None,
) -> list[Pair]:
    """Learn up to ``num_merges`` BPE merges from chunk counts.

    Args:
        chunk_counts: distinct pre-tokenized chunk -> number of occurrences.
        num_merges: how many merges to learn (vocab size = 256 + merges + specials).
        min_frequency: stop early when the best pair occurs fewer times than
            this. A pair seen once is noise, not a pattern worth a token.
        progress: called with the number of merges done after each merge.

    Returns:
        The merges in order. Merge ``i`` creates token id ``256 + i``.
    """
    if num_merges < 0:
        raise ValueError(f"num_merges must be >= 0, got {num_merges}")
    # Sorting makes the chunk ids (and so the whole run) independent of dict order.
    items = sorted((chunk, n) for chunk, n in chunk_counts.items() if n > 0)
    words: list[list[int]] = [list(chunk.encode("utf-8")) for chunk, _ in items]
    freqs: list[int] = [n for _, n in items]

    pair_counts: dict[Pair, int] = {}
    where: dict[Pair, set[int]] = {}
    for i, word in enumerate(words):
        f = freqs[i]
        for pair in zip(word, word[1:], strict=False):
            pair_counts[pair] = pair_counts.get(pair, 0) + f
            where.setdefault(pair, set()).add(i)

    # heapq is a min-heap: store -count, and the pair itself breaks ties (smallest first).
    heap: list[tuple[int, Pair]] = [(-c, p) for p, c in pair_counts.items()]
    heapq.heapify(heap)

    merges: list[Pair] = []
    while len(merges) < num_merges and heap:
        neg_count, best = heapq.heappop(heap)
        if pair_counts.get(best, 0) != -neg_count:
            continue  # stale entry: a fresher one for this pair is (or was) in the heap
        if -neg_count < min_frequency:
            break
        new_id = 256 + len(merges)
        merges.append(best)

        delta: dict[Pair, int] = {}
        for i in where.pop(best):
            old = words[i]
            new = merge_pair(old, best, new_id)
            words[i] = new
            f = freqs[i]
            old_pairs = list(zip(old, old[1:], strict=False))
            new_pairs = list(zip(new, new[1:], strict=False))
            for p in old_pairs:
                delta[p] = delta.get(p, 0) - f
            for p in new_pairs:
                delta[p] = delta.get(p, 0) + f
            old_set, new_set = set(old_pairs), set(new_pairs)
            for p in old_set - new_set:
                chunks = where.get(p)
                if chunks is not None:
                    chunks.discard(i)
            for p in new_set - old_set:
                where.setdefault(p, set()).add(i)

        for p, d in delta.items():
            if d == 0:
                continue
            c = pair_counts.get(p, 0) + d
            if c > 0:
                pair_counts[p] = c
                heapq.heappush(heap, (-c, p))
            else:
                pair_counts.pop(p, None)
                where.pop(p, None)
        if progress is not None:
            progress(len(merges))
    return merges
