# Aupus 0.1 data

What Aupus is trained on, under which licenses, and exactly how raw text
becomes training tokens. Every number here can be reproduced with the commands
at the end.

## Sources

| Source | Used for | License | Status |
|---|---|---|---|
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) `sample-10BT`, revision `87f0914` | Tokenizer training (300 MB), pretraining (~2.5B tokens) | ODC-By 1.0 | Tokenizer sample and M5 sample prepared; full set in M7 |
| [OpenAssistant oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2), revision `179dd21` (English, top-ranked branches) | SFT | Apache-2.0 | Converted: 5,270 conversations |
| [Databricks Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k), revision `bdd27f4` | SFT | CC BY-SA 3.0 | Converted: 15,011 conversations |
| `data/identity.jsonl` (hand-written) | SFT | Apache-2.0 | **To be written** (see below) |
| HellaSwag, ARC-Easy, PIQA | Evaluation only, never training | MIT / CC BY-SA 4.0 / AFL-3.0 | Converted (see below) |

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

## Fine-tuning data (Milestone 10)

`scripts/prepare_sft_data.py` → `aupus/data/sft.py`. It writes
`data/sft/{train,val}.jsonl` (one `{"id", "source", "messages"}` conversation
per line) and `data/sft/stats.json`.

- **oasst2**: message trees written and ranked by volunteers. We keep messages
  that are English, not deleted and passed review. From each root prompt we follow
  the best-ranked reply at every level (prompter follow-ups aren't ranked, so the
  earliest one is taken), and end on an assistant turn. oasst2 already contains
  oasst1: only 8 of oasst1's 88,838 messages are missing from it, so oasst1
  isn't downloaded separately.
- **Identity filter**: a conversation in which the *assistant* calls itself Open
  Assistant or mentions LAION is dropped (130 of 5,401 trees), so Aupus doesn't
  learn another assistant's name.
- **Dolly**: user turn = instruction, plus the reference passage after a blank
  line when there is one. Wikipedia citation marks like `[12]` are removed.
- All text goes through the pretraining `normalize`.
- **Split**: 5% val by hash of the conversation id. Identity conversations all go
  to train, repeated 3× (they are a tiny share of the set).

Measured (2026-11-08, without an identity file yet):

| | Conversations | Train / val | Mean tokens | Cut at 1,024 tokens |
|---|---|---|---|---|
| oasst2 (English top-1 branches) | 5,270 | 5,005 / 265 | 432 | 333 (6%) |
| Dolly 15k | 15,011 | 14,271 / 740 | 169 | 213 (1%); 126 dropped (no reply left) |
| **Total** | 20,281 | 19,276 / 1,005 | | ≈ 4.8M tokens per epoch, 3.2M of them in replies |

A conversation longer than the context window keeps its first 1,024 tokens. The
loss is only on assistant replies and the `<|end|>` that closes each one (chat
template: `aupus/tokenizer/chat.py`).

**Licensing note.** Dolly is CC BY-SA 3.0 (share-alike), the same kind of license
that led us to leave Wikipedia out of pretraining (decision 4). To train without it:
`python scripts/prepare_sft_data.py --sources oasst2,identity`.

### `data/identity.jsonl`

The only training text in git. A few dozen short conversations that tell the
model what it is, so "who are you?" gets a true answer: its name, that it is a
small open-source model built from scratch as a learning project, that it can be
wrong, has no internet access and doesn't remember past chats. **It must be
written by a person**, like all training data here: no model-generated text.
One conversation per line:

```json
{"messages": [{"role": "user", "content": "What is your name?"}, {"role": "assistant", "content": "I'm Aupus, a small language model."}]}
```

Vary the wording of the questions (that is what makes it generalise) and keep the
answers consistent and modest. `prepare_sft_data.py` validates every line.

## Evaluation data (Milestone 9)

`scripts/download_eval_data.py` writes `data/eval/<task>.jsonl` with
`{"id", "context", "choices", "label"}` per question, at pinned revisions. The
prompt formats are lm-evaluation-harness's, so results are comparable with
published numbers (`aupus/eval/tasks.py`).

| Task | Source (revision) | Split | Questions | License |
|---|---|---|---|---|
| HellaSwag | `Rowan/hellaswag` (`218ec52`) | validation | 10,042 | MIT |
| ARC-Easy | `allenai/ai2_arc` (`210d026`) | test | 2,376 | CC BY-SA 4.0 |
| PIQA | `baber/piqa` (`142f6d7`), parquet copy of `ybisk/piqa` | validation | 1,838 | AFL-3.0 |

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

# Milestone 7, the full set: download the pinned parquet files (resumable, SHA-256 checked), then process
python scripts/download_fineweb_edu.py --files 4                 # 8.6 GB, ~2.9M documents
python scripts/prepare_data.py --source parquet --input data/raw/fineweb_edu --out data/pretrain/fineweb_edu

# Milestones 9 and 10: benchmarks (~7 MB) and fine-tuning conversations (~80 MB download)
python scripts/download_eval_data.py
python scripts/prepare_sft_data.py
```
