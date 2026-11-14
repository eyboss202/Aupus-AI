"""Tests for the byte-level BPE tokenizer (pre-tokenization, training, encode/decode, file format)."""

from __future__ import annotations

import json
import random
from collections import Counter

import pytest

from vulpes.tokenizer import (
    ASSISTANT,
    EOS,
    PATTERN,
    SPECIAL_TOKENS,
    USER,
    Tokenizer,
    TokenizerError,
    count_chunks,
    pretokenize,
    train_bpe,
)
from vulpes.tokenizer.train import merge_pair

CORPUS = [
    "The quick brown fox jumps over the lazy dog. The dog sleeps.",
    "In 2024, the fox ate 12,345 berries; the dog ate none!",
    "Don't stop. You're right, I'll go. We've seen it, they'd say.",
    "def train(model, data):\n    for x in data:\n        model.step(x)\n    return model\n",
    "Café naïve résumé, 日本語のテキスト, emoji 😀🚀 and math: x² + y² = z².",
    "the the the then there these those thesis theory therapy thermal",
] * 5


def random_text(rng: random.Random, n: int) -> str:
    """Random strings mixing ASCII, accented letters, CJK, emoji and whitespace (no surrogates)."""
    pools = [
        lambda: chr(rng.randint(0x20, 0x7E)),
        lambda: rng.choice(" \n\t\r  "),
        lambda: chr(rng.randint(0xC0, 0x24F)),
        lambda: chr(rng.randint(0x4E00, 0x9FFF)),
        lambda: chr(rng.randint(0x1F300, 0x1FAFF)),
        lambda: chr(rng.choice([rng.randint(0, 0xD7FF), rng.randint(0xE000, 0x10FFFF)])),
    ]
    return "".join(rng.choice(pools)() for _ in range(n))


# ---------------------------------------------------------------- reference implementations


def naive_train(texts: list[str], num_merges: int, min_frequency: int = 2) -> list[tuple[int, int]]:
    """Textbook BPE: re-count every pair before every merge. Slow but obviously correct."""
    counts = Counter(chunk for t in texts for chunk in pretokenize(t))
    words = {chunk: list(chunk.encode("utf-8")) for chunk in counts}
    merges = []
    for _ in range(num_merges):
        pairs: Counter[tuple[int, int]] = Counter()
        for chunk, ids in words.items():
            for p in zip(ids, ids[1:], strict=False):
                pairs[p] += counts[chunk]
        if not pairs:
            break
        best = min(pairs, key=lambda p: (-pairs[p], p))  # most frequent, then smallest pair
        if pairs[best] < min_frequency:
            break
        new_id = 256 + len(merges)
        merges.append(best)
        words = {c: merge_pair(ids, best, new_id) for c, ids in words.items()}
    return merges


def naive_encode(tok: Tokenizer, text: str) -> list[int]:
    """Replay every merge in training order on every chunk."""
    out = []
    for chunk in pretokenize(text):
        ids = list(chunk.encode("utf-8"))
        for i, pair in enumerate(tok.merges):
            ids = merge_pair(ids, pair, 256 + i)
        out.extend(ids)
    return out


@pytest.fixture(scope="module")
def trained() -> Tokenizer:
    return Tokenizer.train(CORPUS, vocab_size=256 + 150 + len(SPECIAL_TOKENS))


# ---------------------------------------------------------------- pre-tokenization


def test_pretokenize_examples():
    assert pretokenize("Hello world's 2024!") == ["Hello", " world", "'s", " ", "202", "4", "!"]
    assert pretokenize("I DON'T") == ["I", " DON", "'T"]
    assert pretokenize("a  b\n\n  c") == ["a", " ", " b", "\n\n", " ", " c"]
    assert pretokenize("snake_case") == ["snake", "_", "case"]
    assert pretokenize("x):\n    y") == ["x", "):\n", "   ", " y"]


def test_pretokenize_is_lossless_on_random_unicode():
    rng = random.Random(0)
    for _ in range(3000):
        text = random_text(rng, rng.randint(0, 60))
        assert "".join(pretokenize(text)) == text


# ---------------------------------------------------------------- training


def test_merge_pair_is_left_to_right_non_overlapping():
    assert merge_pair([1, 1, 1], (1, 1), 9) == [9, 1]
    assert merge_pair([1, 1, 1, 1], (1, 1), 9) == [9, 9]
    assert merge_pair([1, 2, 3, 1, 2], (1, 2), 9) == [9, 3, 9]
    assert merge_pair([], (1, 2), 9) == []


def test_incremental_training_matches_naive_reference():
    texts = CORPUS + [random_text(random.Random(1), 400)]
    fast = train_bpe(count_chunks(texts), num_merges=300)
    assert fast == naive_train(texts, num_merges=300)


def test_repeated_byte_runs_match_naive_reference():
    # Overlapping pairs (a, a) are the classic place for incremental counting bugs.
    texts = ["aaaaaaa aaaa aaa aa a", "bbbbbbbbbb ababab aaabbb", "        x"] * 3
    assert train_bpe(count_chunks(texts), 60) == naive_train(texts, 60)


def test_first_merge_is_the_most_frequent_pair():
    merges = train_bpe(Counter({"ab": 5, "cd": 3, "abcd": 1}), num_merges=1)
    assert merges == [(ord("a"), ord("b"))]


def test_ties_break_on_smallest_pair():
    merges = train_bpe(Counter({"zy": 4, "ba": 4}), num_merges=2)
    assert merges == [(ord("b"), ord("a")), (ord("z"), ord("y"))]


def test_training_stops_at_min_frequency():
    merges = train_bpe(Counter({"abc": 1}), num_merges=10, min_frequency=2)
    assert merges == []


def test_parallel_counting_matches_serial():
    assert count_chunks(CORPUS, num_workers=2, batch_size=4) == count_chunks(CORPUS)


def test_training_is_deterministic_and_hits_vocab_size(trained):
    again = Tokenizer.train(CORPUS, vocab_size=trained.vocab_size)
    assert again.merges == trained.merges
    assert trained.vocab_size == 256 + 150 + len(SPECIAL_TOKENS)


def test_vocab_too_small_is_rejected():
    with pytest.raises(ValueError, match="smaller than 256"):
        Tokenizer.train(CORPUS, vocab_size=100)


# ---------------------------------------------------------------- encode / decode


def test_byte_fallback_roundtrips_anything():
    tok = Tokenizer()  # no merges: pure bytes
    rng = random.Random(2)
    for _ in range(200):
        text = random_text(rng, rng.randint(0, 80))
        ids = tok.encode(text)
        assert ids == list(text.encode("utf-8"))
        assert tok.decode(ids) == text


def test_roundtrip_and_compression(trained):
    rng = random.Random(3)
    for text in [*CORPUS, "", " ", "\n", "zzz unseen wörds 🦊", *(random_text(rng, 100) for _ in range(100))]:
        assert trained.decode(trained.encode(text)) == text
    sample = CORPUS[0]
    assert len(trained.encode(sample)) < len(sample.encode("utf-8")) / 2


def test_encode_matches_replaying_merges_in_order(trained):
    rng = random.Random(4)
    for text in [*CORPUS, *(random_text(rng, 60) for _ in range(50))]:
        assert trained.encode(text) == naive_encode(trained, text)


def test_encode_is_stable_with_cache(trained):
    text = " ".join(CORPUS)
    first = trained.encode(text)
    assert trained.encode(text) == first


def test_decode_partial_utf8_uses_replacement_char():
    tok = Tokenizer()
    ids = tok.encode("😀")
    assert len(ids) == 4
    assert tok.decode(ids[:2]) == "�"
    assert tok.decode_bytes(ids[:2]) == "😀".encode()[:2]


def test_decode_rejects_unknown_ids(trained):
    with pytest.raises(TokenizerError, match="out of range"):
        trained.decode([trained.vocab_size])
    with pytest.raises(TokenizerError, match="out of range"):
        trained.decode([-1])


# ---------------------------------------------------------------- special tokens


def test_special_tokens_take_the_last_ids(trained):
    n = trained.vocab_size
    assert list(trained.special_tokens) == list(SPECIAL_TOKENS)
    assert sorted(trained.special_tokens.values()) == list(range(n - len(SPECIAL_TOKENS), n))


def test_special_tokens_are_plain_text_unless_allowed(trained):
    text = f"{USER}hi{EOS}"
    plain = trained.encode(text)
    assert not any(trained.is_special(i) for i in plain)
    assert trained.decode(plain) == text

    ids = trained.encode(text, allow_special=True)
    assert ids[0] == trained.special_id(USER)
    assert ids[-1] == trained.special_id(EOS)
    assert trained.decode(ids) == text
    assert trained.decode(ids, skip_special=True) == "hi"


def test_adjacent_and_repeated_special_tokens(trained):
    ids = trained.encode(f"{ASSISTANT}{ASSISTANT}x{EOS}", allow_special=True)
    a, e = trained.special_id(ASSISTANT), trained.special_id(EOS)
    assert ids[:2] == [a, a] and ids[-1] == e


def test_special_id_unknown():
    with pytest.raises(TokenizerError, match="not a special token"):
        Tokenizer().special_id("<|nope|>")


def test_duplicate_special_tokens_rejected():
    with pytest.raises(TokenizerError, match="unique"):
        Tokenizer(special_tokens=["<|a|>", "<|a|>"])


# ---------------------------------------------------------------- file format


def test_save_load_roundtrip(trained, tmp_path):
    path = tmp_path / "tok.json"
    trained.save(path)
    loaded = Tokenizer.load(path)
    assert loaded.merges == trained.merges
    assert loaded.special_tokens == trained.special_tokens
    assert loaded.pattern == PATTERN
    for text in CORPUS:
        assert loaded.encode(text, allow_special=True) == trained.encode(text, allow_special=True)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["format"] == "vulpes-bpe" and data["vocab_size"] == trained.vocab_size


def test_save_empty_tokenizer(tmp_path):
    path = tmp_path / "empty.json"
    Tokenizer().save(path)
    assert Tokenizer.load(path).vocab_size == 256 + len(SPECIAL_TOKENS)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d.update(format="other"), "not a vulpes-bpe"),
        (lambda d: d.update(version=99), "unsupported tokenizer version"),
        (lambda d: d.update(extra=1), "keys must be exactly"),
        (lambda d: d.update(merges=[[1, 2, 3]]), "merges must be"),
        (lambda d: d.update(merges=[[1, 9999]]), "doesn't exist yet"),
        (lambda d: d.update(vocab_size=5), "doesn't match"),
        (lambda d: d["special_tokens"].update({EOS: 0}), "directly follow"),
    ],
)
def test_load_rejects_malformed_files(trained, tmp_path, mutate, message):
    d = trained.to_dict()
    mutate(d)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(TokenizerError, match=message):
        Tokenizer.load(path)


def test_dump_vocab(trained, tmp_path):
    path = tmp_path / "vocab.txt"
    trained.dump_vocab(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == trained.vocab_size
    assert lines[ord("A")] == f"{ord('A')}\t'A'"
    assert lines[-1].endswith("(special)")
