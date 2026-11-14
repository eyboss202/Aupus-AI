# Vulpes 0.1 data

What Vulpes is trained on, under which licenses, and exactly how raw text
becomes training tokens. Every number here can be reproduced with the commands
at the end.

## Sources

| Source | Used for | License | Status |
|---|---|---|---|
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) `sample-10BT`, revision `87f0914` | Tokenizer training (300 MB), pretraining (~2.5B tokens) | ODC-By 1.0 | Prepared: 2.94B train + 14.6M val tokens (M7) |
| [OpenAssistant oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2), revision `179dd21` (English, top-ranked branches) | SFT | Apache-2.0 | Converted: 5,270 conversations |
| [Databricks Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k), revision `bdd27f4` | SFT | CC BY-SA 3.0 | Converted: 15,011 conversations |
| `data/identity.jsonl` (questions hand-written; replies AI-assisted, see below) | SFT | Apache-2.0 | 44 conversations (2026-11-10) |
| HellaSwag, ARC-Easy, PIQA | Evaluation only, never training | MIT / CC BY-SA 4.0 / AFL-3.0 | Converted (see below) |

Only human-written, openly licensed text is used. No LLM-generated instruction
data and no other model's outputs. **One declared exception:** most replies in the small
identity set were corrected with an AI assistant's help (see `data/identity.jsonl` below). English Wikipedia was considered and
declined for 0.1 (share-alike licensing, see `docs/PLAN.md` §10).

FineWeb-Edu is Common Crawl web text that Hugging Face filtered for
educational value with a classifier, plus language ID, quality filters and
per-crawl dedup. Our pipeline is a second, lighter pass on top of that.

## Pretraining pipeline

`scripts/prepare_data.py` → `vulpes/data/pipeline.py`. Settings live in
`configs/data/fineweb_edu.yaml` (strict: a typo in a key is an error). The
resolved settings are copied into each dataset's `meta.json`.

```
raw document
  → normalize        NFC, \r\n → \n, drop invisible/control chars, trim trailing spaces, ≤ 1 blank line
  → quality filters  first failing rule is recorded as the drop reason
  → exact dedup      BLAKE2b-128 of the normalised text; keep first
  → near dedup       MinHash (128 perms, word 5-grams) + LSH (16 bands × 8 rows); keep first
  → split            val iff hash < 0.5% of hash space (stable for any order / worker count)
  → tokenize         Vulpes BPE 32k, special-token strings stay plain text, + <|eos|>
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

## Milestone 7 run: the full pretraining set

The first 4 parquet files of `sample-10BT` (8.61 GB, SHA-256 checked), processed on
the desktop with 3 workers in 68 min (2026-11-09):

```bash
python scripts/download_fineweb_edu.py --files 4
python scripts/prepare_data.py --source parquet --input data/raw/fineweb_edu --out data/pretrain/fineweb_edu --workers 3
```

Results are in `data/pretrain/fineweb_edu/stats.json`:

| | Documents | Share |
|---|---|---|
| Input | 2,916,000 | 100% |
| Dropped by filters | 15,779 | 0.54% |
| — `ellipsis_lines` | 6,123 | 0.21% |
| — `bullet_lines` | 4,684 | 0.16% |
| — `alpha_words` | 3,018 | 0.10% |
| — `too_short` | 1,151 | 0.04% |
| — `replacement_chars` | 560 | 0.02% |
| — other rules (`stop_words` 141, `mean_word_length` 51, `symbol_ratio` 27, `duplicate_lines` 24) | 243 | 0.01% |
| Exact duplicates | 33,732 | 1.16% |
| Near duplicates | 21,692 | 0.74% |
| **Kept** | **2,844,797** | **97.56%** (2,830,585 train / 14,212 val) |

| | |
|---|---|
| Tokens | **2,936,424,290 train** (30 shards), **14,644,688 val** (4.60 bytes/token) |
| Throughput | 4,062 s for 13.9 GB = 3.4 MB/s (718 docs/s) on 3 workers (i5-9400F) |
| Output | 5.5 GB of shards |

The filters drop about the same share as on the M5 sample, with the same rules on
top. Duplicates are another story: 1.9% of documents in total, against 0.03% in
the 300 MB sample, because copies across crawls only show up at this scale.

The 2.94B tokens are more than the 2.5B budget for `small` (0.85 epochs), so a
2.5B-token run never sees the same window twice.

## Fine-tuning data (Milestone 10)

`scripts/prepare_sft_data.py` → `vulpes/data/sft.py`. It writes
`data/sft/{train,val}.jsonl` (one `{"id", "source", "messages"}` conversation
per line) and `data/sft/stats.json`.

- **oasst2**: message trees written and ranked by volunteers. We keep messages
  that are English, not deleted and passed review. From each root prompt we follow
  the best-ranked reply at every level (prompter follow-ups aren't ranked, so the
  earliest one is taken), and end on an assistant turn. oasst2 already contains
  oasst1: only 8 of oasst1's 88,838 messages are missing from it, so oasst1
  isn't downloaded separately.
- **Identity filter**: a conversation in which the *assistant* calls itself Open
  Assistant or mentions LAION is dropped (130 of 5,401 trees), so Vulpes doesn't
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
template: `vulpes/tokenizer/chat.py`).

**Licensing note.** Dolly is CC BY-SA 3.0 (share-alike), the same kind of license
that led us to leave Wikipedia out of pretraining (decision 4). To train without it:
`python scripts/prepare_sft_data.py --sources oasst2,identity`.

### `data/identity.jsonl`

The only training text in git. A few dozen short conversations that tell the
model what it is, so "who are you?" gets a true answer: its name, that it is a
small open-source model built from scratch as a learning project, that it can be
wrong, has no internet access and doesn't remember past chats. The rule was that it is
written by a person, like all training data here.

**Declared exception (2026-11-10).** The 44 user questions and one reply were written by
the project author. The author's first replies described a generic large assistant (coding,
translation, many languages, cloud servers) and never used the name Vulpes, so at the
author's request an AI assistant (Claude, used to develop this project) rewrote 43 of the
44 replies to state the true facts of `docs/IDENTITY.md`, keeping the author's wording
where it was already true. Those replies are therefore AI-written text: about 1,100 tokens,
repeated 3× in fine-tuning, next to ~4.5M tokens of human-written conversations.
One conversation per line:

```json
{"messages": [{"role": "user", "content": "What is your name?"}, {"role": "assistant", "content": "I'm Vulpes, a small language model."}]}
```

Vary the wording of the questions (that is what makes it generalise) and keep the
answers consistent and modest. **[`docs/IDENTITY.md`](IDENTITY.md) is the writing guide**: the facts
to convey, the topics to cover and the style. `python scripts/check_identity.py` checks the file
(format, length, duplicates, personal details, non-English text); `prepare_sft_data.py` validates
every line again.

## Evaluation data (Milestone 9)

`scripts/download_eval_data.py` writes `data/eval/<task>.jsonl` with
`{"id", "context", "choices", "label"}` per question, at pinned revisions. The
prompt formats are lm-evaluation-harness's, so results are comparable with
published numbers (`vulpes/eval/tasks.py`).

| Task | Source (revision) | Split | Questions | License |
|---|---|---|---|---|
| HellaSwag | `Rowan/hellaswag` (`218ec52`) | validation | 10,042 | MIT |
| ARC-Easy | `allenai/ai2_arc` (`210d026`) | test | 2,376 | CC BY-SA 4.0 |
| PIQA | `baber/piqa` (`142f6d7`), parquet copy of `ybisk/piqa` | validation | 1,838 | AFL-3.0 |

## Dataset format

```
data/pretrain/<name>/
  meta.json          {"format": "vulpes-shards", "version": 1, "dtype": "uint16",
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

`vulpes.data.PretrainLoader` memory-maps the shards and serves
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
