"""Tests for the data pipeline: normalisation, filters, dedup, split, shards, loader, end to end."""

from __future__ import annotations

import json
import random

import numpy as np
import pytest
import torch

from aupus.config import ConfigError, load_config
from aupus.data import (
    DataConfig,
    DedupConfig,
    Deduplicator,
    FilterConfig,
    MinHasher,
    PretrainLoader,
    ShardWriter,
    check,
    check_tokenizer,
    estimated_jaccard,
    eval_batches,
    exact_hash,
    is_val,
    iter_parquet,
    normalize,
    open_split,
    prepare,
)
from aupus.data.dedup import shingles
from aupus.data.shards import write_meta
from aupus.tokenizer import EOS, Tokenizer

FILTERS = FilterConfig()

WORDS = [
    "river",
    "mountain",
    "forest",
    "teacher",
    "student",
    "lesson",
    "energy",
    "water",
    "light",
    "planet",
    "history",
    "science",
    "language",
    "garden",
    "market",
    "winter",
    "summer",
    "animal",
    "plant",
    "ocean",
    "city",
    "village",
    "bridge",
    "music",
    "story",
]


def prose(rng: random.Random, n_sentences: int = 12) -> str:
    """Random but English-looking prose that passes every filter."""
    sentences = []
    for _ in range(n_sentences):
        body = " ".join(rng.choice(WORDS) for _ in range(rng.randint(6, 12)))
        sentences.append(f"The {body} and that is the way to learn with care.")
    return " ".join(sentences)


# ---------------------------------------------------------------- normalisation


def test_normalize():
    assert normalize("café") == "café"  # NFD -> NFC
    assert normalize("a\r\nb\rc d") == "a\nb\nc\nd"
    assert normalize("in­visible​chars﻿\x07") == "invisiblechars"
    assert normalize("trailing   \nspaces\t\n") == "trailing\nspaces"
    assert normalize("a\n\n\n\n\nb") == "a\n\nb"
    assert normalize("  padded  ") == "padded"
    assert normalize("tab\tkept") == "tab\tkept"
    assert normalize("👨‍👩‍👧") == "👨‍👩‍👧"  # zero-width joiner kept for emoji


def test_normalize_is_idempotent():
    rng = random.Random(0)
    pool = "ab \n\r\t­​́é. "
    for _ in range(500):
        text = "".join(rng.choice(pool) for _ in range(rng.randint(0, 40)))
        once = normalize(text)
        assert normalize(once) == once


# ---------------------------------------------------------------- filters


def test_good_prose_passes():
    assert check(prose(random.Random(1)), FILTERS) is None


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("", "empty"),
        ("the and of " * 30 + "�" * 10, "replacement_chars"),
        ("The cat sat on the mat and that was it.", "too_short"),
        ("the and " + "a b c d e " * 20, "mean_word_length"),
        ("the and with " + "supercalifragilistic " * 60, "mean_word_length"),
        ("the and with " + "#tag word " * 40, "symbol_ratio"),
        ("the and with " + "1234 5678 9012 3456 7890 " * 20, "alpha_words"),
        (" ".join(["cheap shoes buy online discount sale"] * 20), "stop_words"),
        (
            "\n".join(
                ["The cat and the dog went to the park and then..."] * 2 + ["the end of the day was fine"]
            )
            * 10,
            "ellipsis_lines",
        ),
        ("\n".join(["- the item with a long description of things"] * 20), "bullet_lines"),
        (
            "\n".join(
                ["Click here to subscribe to the newsletter and win"] * 5 + [prose(random.Random(2), 1)]
            ),
            "duplicate_lines",
        ),
        (prose(random.Random(3)) + " Lorem ipsum dolor sit amet.", "banned_phrase"),
    ],
)
def test_filter_rules(text, reason):
    assert check(text, FILTERS) == reason


def test_too_long():
    assert check(prose(random.Random(4), 20), FilterConfig(max_words=100)) == "too_long"


# ---------------------------------------------------------------- dedup


def test_shingles_and_signature_are_deterministic():
    text = prose(random.Random(5))
    # Set iteration order varies between processes (str hashing is salted); the *set* of
    # shingle hashes must not, and MinHash only takes minimums, so order never matters.
    words = text.lower().replace(".", " ").split()
    distinct_grams = {" ".join(words[i : i + 5]) for i in range(len(words) - 4)}
    assert len(shingles(text, 5)) == len(distinct_grams)
    assert set(shingles(text, 5).tolist()) == set(shingles(text, 5).tolist())
    h1, h2 = MinHasher(DedupConfig()), MinHasher(DedupConfig())
    assert np.array_equal(h1.signature(text), h2.signature(text))
    assert h1.signature(text).shape == (128,)


def test_minhash_estimates_jaccard():
    words = [f"w{i}" for i in range(400)]
    a = words[:300]
    b = words[100:]  # 5-grams: 296 in each, 196 shared -> true Jaccard = 196 / 396 ~ 0.495
    hasher = MinHasher(DedupConfig(num_perm=256, bands=32))
    est = estimated_jaccard(hasher.signature(" ".join(a)), hasher.signature(" ".join(b)))
    assert est == pytest.approx(196 / 396, abs=0.08)


def test_long_documents_are_processed_in_chunks():
    # More shingles than one NumPy block: the result must equal a single-block computation.
    text = " ".join(f"tok{i}" for i in range(10_000))
    hasher = MinHasher(DedupConfig())
    x = shingles(text, 5)
    expected = ((hasher._a * x[None, :] + hasher._b) % np.uint64(4294967291)).min(axis=1).astype(np.uint32)
    assert np.array_equal(hasher.signature(text), expected)


def test_near_duplicates_are_caught_and_distinct_docs_kept():
    rng = random.Random(7)
    hasher = MinHasher(DedupConfig())
    dedup = Deduplicator()
    base = prose(rng, 40)
    near = base.replace("The", "A", 1) + " One extra closing sentence here."
    other = prose(rng, 40)
    assert not dedup.is_near_duplicate(hasher.band_keys(hasher.signature(base)))
    assert dedup.is_near_duplicate(hasher.band_keys(hasher.signature(near)))
    assert not dedup.is_near_duplicate(hasher.band_keys(hasher.signature(other)))


def test_exact_dedup():
    dedup = Deduplicator()
    assert not dedup.is_exact_duplicate(exact_hash("same text"))
    assert dedup.is_exact_duplicate(exact_hash("same text"))
    assert not dedup.is_exact_duplicate(exact_hash("other text"))


def test_dedup_config_validation():
    with pytest.raises(ValueError, match="divisible"):
        DedupConfig(num_perm=100, bands=16)
    assert DedupConfig().rows == 8


# ---------------------------------------------------------------- split and shards


def test_val_split_is_deterministic_and_close_to_fraction():
    digests = [exact_hash(f"doc {i}") for i in range(40_000)]
    flags = [is_val(d, 0.005) for d in digests]
    assert flags == [is_val(d, 0.005) for d in digests]
    assert 0.004 < sum(flags) / len(flags) < 0.006
    assert not any(is_val(d, 0.0) for d in digests[:100])


@pytest.fixture(scope="module")
def tokenizer_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("tok") / "tok.json"
    rng = random.Random(8)
    Tokenizer.train([prose(rng) for _ in range(30)], vocab_size=256 + 200 + 7).save(path)
    return path


def write_dataset(path, tokenizer_path, train_docs, val_docs=(), shard_tokens=1000):
    splits = {}
    for split, docs in (("train", train_docs), ("val", val_docs)):
        writer = ShardWriter(path, split, shard_tokens)
        for d in docs:
            writer.add(np.asarray(d))
        splits[split] = writer.close()
    write_meta(path, splits, tokenizer_path, vocab_size=463, eos_id=457)


def test_shard_roundtrip(tmp_path, tokenizer_path):
    docs = [np.arange(i * 100, i * 100 + 50) for i in range(30)]  # 1500 tokens -> 2 shards of 1000 limit
    write_dataset(tmp_path, tokenizer_path, docs)
    meta, shards = open_split(tmp_path, "train")
    assert len(shards) == 2
    assert np.array_equal(np.concatenate([s.tokens for s in shards]), np.concatenate(docs))
    assert meta["splits"]["train"][0]["tokens"] == 1000
    check_tokenizer(meta, tokenizer_path)


def test_open_split_detects_problems(tmp_path, tokenizer_path):
    write_dataset(tmp_path, tokenizer_path, [np.arange(100)])
    with pytest.raises(ValueError, match="no split"):
        open_split(tmp_path, "test")
    (tmp_path / "train_00000.bin").write_bytes(b"\0" * 10)
    with pytest.raises(ValueError, match="expected 100 tokens"):
        open_split(tmp_path, "train")
    other = tmp_path / "other.json"
    Tokenizer().save(other)
    meta = json.loads((tmp_path / "meta.json").read_text())
    with pytest.raises(ValueError, match="sha256"):
        check_tokenizer(meta, other)


# ---------------------------------------------------------------- loader


@pytest.fixture
def stream_dir(tmp_path, tokenizer_path):
    # Token value == position, so windows are easy to check. Three 1000-token documents;
    # the writer never splits a document, so each one fills (and flushes) its own shard.
    write_dataset(
        tmp_path,
        tokenizer_path,
        [np.arange(i * 1000, (i + 1) * 1000) for i in range(3)],
        [np.arange(250)],
        shard_tokens=1000,
    )
    return tmp_path


def test_loader_windows_and_shift(stream_dir):
    loader = PretrainLoader(stream_dir, "train", batch_size=4, seq_len=99)
    # (1000 - 1) // 99 = 10 windows per shard, 3 shards.
    assert loader.num_windows == 30
    x, y = next(loader)
    assert x.shape == y.shape == (4, 99) and x.dtype == torch.int64
    assert torch.equal(x[:, 1:], y[:, :-1])
    assert torch.all((x[:, 0] % 1000) % 99 == 0)  # windows start on seq_len boundaries within a shard


def test_loader_covers_every_window_once_per_epoch(stream_dir):
    loader = PretrainLoader(stream_dir, "train", batch_size=5, seq_len=99, seed=3)
    starts = []
    for _ in range(6):  # 6 x 5 = 30 windows = one epoch
        x, _ = next(loader)
        starts += x[:, 0].tolist()
    assert len(set(starts)) == 30
    assert loader.epoch == 0 and loader.position == 30
    next(loader)
    assert loader.epoch == 1


def test_loader_is_deterministic_and_resumable(stream_dir):
    a = PretrainLoader(stream_dir, "train", batch_size=4, seq_len=99, seed=1)
    for _ in range(9):  # crosses the epoch boundary mid-batch (30 windows / 4)
        next(a)
    state = a.state_dict()
    expected = [next(a)[0] for _ in range(5)]

    b = PretrainLoader(stream_dir, "train", batch_size=4, seq_len=99, seed=1)
    b.load_state_dict(state)
    for want in expected:
        assert torch.equal(next(b)[0], want)

    c = PretrainLoader(stream_dir, "train", batch_size=4, seq_len=99, seed=2)
    assert not torch.equal(next(c)[0], next(PretrainLoader(stream_dir, "train", 4, 99, seed=1))[0])
    with pytest.raises(ValueError, match="mismatch for seed"):
        c.load_state_dict(state)


def test_loader_resumes_with_another_batch_size(stream_dir):
    # Moving a run to a machine with less memory: the same windows in the same order,
    # only cut into smaller (or bigger) batches, also across the epoch boundary.
    a = PretrainLoader(stream_dir, "train", batch_size=4, seq_len=99, seed=1)
    for _ in range(6):
        next(a)
    state = a.state_dict()
    expected = torch.cat([next(a)[0] for _ in range(5)])  # windows 24..43: crosses into epoch 1

    for batch_size in (1, 2, 5):
        b = PretrainLoader(stream_dir, "train", batch_size=batch_size, seq_len=99, seed=1)
        b.load_state_dict(state)
        got = torch.cat([next(b)[0] for _ in range(len(expected) // batch_size)])
        assert torch.equal(got, expected)
        assert (b.epoch, b.position) == (a.epoch, a.position)


def test_eval_batches_is_ordered_and_finite(stream_dir):
    batches = list(eval_batches(stream_dir, "val", batch_size=2, seq_len=50))
    # 250 tokens -> (250 - 1) // 50 = 4 windows -> 2 batches
    assert len(batches) == 2
    assert batches[0][0][0, 0].item() == 0 and batches[0][0][1, 0].item() == 50
    assert len(list(eval_batches(stream_dir, "val", batch_size=1, seq_len=50, max_batches=3))) == 3


def test_loader_rejects_split_without_full_window(stream_dir):
    with pytest.raises(ValueError, match="no complete window"):
        PretrainLoader(stream_dir, "val", batch_size=1, seq_len=500)


# ---------------------------------------------------------------- end to end


def make_corpus() -> list[str]:
    rng = random.Random(9)
    docs = [prose(rng, 15) for _ in range(300)]
    docs += docs[:20]  # exact duplicates
    docs += [d.replace("The", "A", 1) + " And one more line." for d in docs[20:40]]  # near duplicates
    docs += ["too short"] * 5 + [" ".join(["cheap shoes buy online"] * 30)]  # filtered
    rng.shuffle(docs)
    return docs


def test_prepare_end_to_end(tmp_path, tokenizer_path):
    cfg = DataConfig(val_fraction=0.1, shard_tokens=20_000)
    stats = prepare(make_corpus(), tmp_path, tokenizer_path, cfg)
    docs = stats["documents"]
    assert docs["input"] == 346
    assert docs["filtered:too_short"] == 5 and docs["filtered:stop_words"] == 1
    assert docs["dedup:exact"] == 20
    assert docs["dedup:near"] == 20
    assert docs["kept:train"] + docs["kept:val"] == 300
    assert 10 < docs["kept:val"] < 60

    tok = Tokenizer.load(tokenizer_path)
    meta, shards = open_split(tmp_path, "train")
    stream = np.concatenate([s.tokens for s in shards])
    assert len(stream) == stats["tokens"]["train"]
    assert int((stream == tok.special_id(EOS)).sum()) == docs["kept:train"]  # one <|eos|> per document
    first_doc = stream[: int(np.argmax(stream == tok.special_id(EOS)))]
    assert check(tok.decode(first_doc.tolist()), cfg.filters) is None
    assert json.loads((tmp_path / "stats.json").read_text()) == stats
    assert meta["config"]["val_fraction"] == 0.1


def test_prepare_is_identical_with_workers(tmp_path, tokenizer_path):
    corpus = make_corpus()
    cfg = DataConfig(val_fraction=0.1)
    prepare(corpus, tmp_path / "one", tokenizer_path, cfg, num_workers=1)
    prepare(corpus, tmp_path / "two", tokenizer_path, cfg, num_workers=2, batch_size=16)
    for split in ("train", "val"):
        a = (tmp_path / "one" / f"{split}_00000.bin").read_bytes()
        b = (tmp_path / "two" / f"{split}_00000.bin").read_bytes()
        assert a == b


def test_parquet_source_matches_in_memory(tmp_path, tokenizer_path):
    """Local parquet files give the same documents, in the same order, as the list they came from."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    corpus = make_corpus()
    # Two files, written out of name order, with an extra column and small row groups.
    for name, part in (("001_00000.parquet", corpus[200:]), ("000_00000.parquet", corpus[:200])):
        table = pa.table({"id": list(range(len(part))), "text": part})
        pq.write_table(table, tmp_path / name, row_group_size=37)
    paths = sorted(tmp_path.glob("*.parquet"))
    assert list(iter_parquet(paths, batch_rows=10)) == corpus

    cfg = DataConfig(val_fraction=0.1)
    prepare(corpus, tmp_path / "list", tokenizer_path, cfg)
    prepare(iter_parquet(paths), tmp_path / "parquet", tokenizer_path, cfg)
    for split in ("train", "val"):
        a = (tmp_path / "list" / f"{split}_00000.bin").read_bytes()
        assert a == (tmp_path / "parquet" / f"{split}_00000.bin").read_bytes()


def test_shipped_data_config_loads():
    cfg = load_config(DataConfig, "configs/data/fineweb_edu.yaml")
    assert cfg == DataConfig()
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(DataConfig, "configs/data/fineweb_edu.yaml", overrides={"val_fration": 0.1})
