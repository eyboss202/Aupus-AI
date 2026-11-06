# Aupus 0.1 data

What Aupus is trained on, under which licenses, and exactly how raw text
becomes training tokens. Every number here can be reproduced with the commands
at the end.

## Sources

| Source | Used for | License | Status |
|---|---|---|---|
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) `sample-10BT`, revision `87f0914` | Tokenizer training (300 MB), pretraining (~2.5B tokens) | ODC-By 1.0 | Tokenizer sample and M5 sample prepared; full set in M7 |
| OpenAssistant oasst1 + oasst2 (English, top-ranked branches) | SFT | Apache-2.0 | Milestone 10 |
| Databricks Dolly 15k | SFT | CC BY-SA 3.0 | Milestone 10 |
| `data/identity.jsonl` (hand-written) | SFT | Apache-2.0 | Milestone 10 |

Only human-written, openly licensed text is used. No LLM-generated instruction
data and no other model's outputs. English Wikipedia was considered and
declined for 0.1 (share-alike licensing, see `docs/PLAN.md` §10).

FineWeb-Edu is Common Crawl web text that Hugging Face filtered for
educational value with a classifier, plus language ID, quality filters and
per-crawl dedup. Our pipeline is a second, lighter pass on top of that.

## Pretraining pipeline

`scripts/prepare_data.py` → `aupus/data/pipeline.py`. Settings live in
`configs/data/fineweb_edu.yaml` (strict: a typo in a key is an error). The
resolved settings are copied into each dataset's `meta.json`.

```
raw document
  → normalize        NFC, \r\n → \n, drop invisible/control chars, trim trailing spaces, ≤ 1 blank line
  → quality filters  first failing rule is recorded as the drop reason
  → exact dedup      BLAKE2b-128 of the normalised text; keep first
  → near dedup       MinHash (128 perms, word 5-grams) + LSH (16 bands × 8 rows); keep first
  → split            val iff hash < 0.5% of hash space (stable for any order / worker count)
  → tokenize         Aupus BPE 32k, special-token strings stay plain text, + <|eos|>
  → shards           train_NNNNN.bin / val_NNNNN.bin: raw little-endian uint16, ~100M tokens each
```

Per-document work runs in worker processes. Dedup, split and writing run
in order in the main process, so the output is byte-identical for any number of
workers (tested).

### Quality filters

Gopher rules (Rae et al. 2021), the same family FineWeb uses. A document is
dropped if:

| Rule | Drops if | Catches |
|---|---|---|
| `replacement_chars` | > 0.1% of characters are U+FFFD | text decoded with the wrong encoding upstream |
| `too_short` / `too_long` | < 50 or > 100,000 words | stubs, link pages / data dumps |
| `mean_word_length` | mean word length outside 3–10 | symbol soup, concatenated words, URLs |
| `symbol_ratio` | > 0.1 `#` or `…` per word | hashtag spam, truncated listings |
| `alpha_words` | < 80% of words contain a letter | tables of numbers, code dumps |
| `stop_words` | < 2 of {the, be, to, of, and, that, have, with} | keyword lists, non-English text |
| `ellipsis_lines` | > 30% of lines end in `...` / `…` | search-result snippets, teasers |
| `bullet_lines` | > 90% of lines start with a bullet | navigation menus, link lists |
| `duplicate_lines` | > 30% of lines repeat an earlier line | templated boilerplate |
| `banned_phrase` | contains "lorem ipsum" | placeholder pages |

Manual review of the drops (Milestone 5): the `ellipsis_lines`,
`alpha_words`, `too_short` and `replacement_chars` drops are junk. About
half of the `bullet_lines` drops are actually fine educational lists. That
costs ~0.2% of documents, and we keep the standard threshold anyway rather
than tune rules by hand.

### Near-duplicate detection

Two documents' similarity is the Jaccard index of their sets of word 5-grams.
MinHash estimates it from 128 numbers per document. LSH makes two documents
candidates when all 8 values of any of the 16 bands match. A pair with
similarity 0.8 is caught 95% of the time, 0.5 only 6% (`1 − (1 − s⁸)¹⁶`).
Memory is ~1.1 KB per kept document (≈ 3 GB for 2.7M documents).

## Milestone 5 run: the 300 MB sample

`python scripts/prepare_data.py --input data/tokenizer_sample/train.jsonl --out data/pretrain/sample`
(the first 61,842 FineWeb-Edu documents, the same text the tokenizer was trained
on). Results are in `data/pretrain/sample/stats.json`:

| | Documents | Share |
|---|---|---|
| Input | 61,842 | 100% |
| Dropped by filters | 381 | 0.62% |
| — `ellipsis_lines` | 157 | 0.25% |
| — `bullet_lines` | 119 | 0.19% |
| — `alpha_words` | 66 | 0.11% |
| — `too_short` | 24 | 0.04% |
| — other rules | 15 | 0.02% |
| Exact duplicates | 4 | 0.01% |
| Near duplicates | 12 | 0.02% |
| **Kept** | **61,445** | **99.36%** (61,137 train / 308 val) |

| | |
|---|---|
| Tokens | 64.8M train, 0.33M val (4.6 bytes/token) |
| Throughput | 49 s for 300 MB = **6.1 MB/s** (1,250 docs/s) on 6 processes (i5-9400F) |
| Output | 130 MB of shards |

Few duplicates were found because FineWeb is already deduplicated within each
crawl. Our pass mainly catches copies *across* crawls, and those only show up
in larger samples.

**Projection for Milestone 7** (~2.5B tokens ≈ 11.5 GB of raw text): ~32 min
of processing at this rate, plus the download time. Peak RAM is the dedup set
(~3 GB) plus workers (~0.3 GB each).

## Dataset format

```
data/pretrain/<name>/
  meta.json          {"format": "aupus-shards", "version": 1, "dtype": "uint16",
                      "tokenizer": ..., "tokenizer_sha256": ..., "vocab_size": 32000,
                      "eos_id": 31994, "splits": {"train": [{"file", "tokens"}, ...], "val": [...]},
                      "config": {...resolved DataConfig...}}
  stats.json         counts per drop reason, tokens, throughput
  train_00000.bin    raw uint16 token ids, documents separated by <|eos|>
  val_00000.bin
```

Training checks `tokenizer_sha256` against the tokenizer it loads and refuses
to run on a mismatch.

## Dataloader

`aupus.data.PretrainLoader` memory-maps the shards and serves
`(inputs, targets)` batches of `seq_len` tokens:

- each shard is cut into windows of `seq_len + 1` tokens, so every token is
  predicted once per epoch. Windows pack across document boundaries (separated
  by `<|eos|>`), so no padding is wasted;
- the window order for each epoch is a seeded permutation, so the loader's
  entire state is `(epoch, position)`. That state is saved in checkpoints, and
  resuming continues with exactly the next batch;
- `eval_batches` makes one fixed, ordered pass over `val`, so validation loss
  is comparable across a run.

## Reproducing

```bash
pip install -e ".[data]"
python scripts/download_tokenizer_sample.py                  # 300 MB + 10 MB, pinned revision
python scripts/prepare_data.py --input data/tokenizer_sample/train.jsonl --out data/pretrain/sample
```
