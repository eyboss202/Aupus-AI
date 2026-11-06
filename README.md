# Aupus AI

**Aupus 0.1** is a small English language model that writes text and answers simple
questions, built **completely from scratch**: tokenizer, transformer, training loop,
fine-tuning and inference engine are all written in this repository with plain
PyTorch. There are no pretrained weights, no external model APIs and no high-level
model libraries.

It is also a learning project. The code is meant to be read: docstrings explain
*why* each piece exists, not just what it does.

> **Status: pre-alpha, under construction.** The tokenizer, model, data pipeline and
> pretraining loop are written and tested, and a small `tiny` model has been trained end to
> end as a check. Aupus 0.1 itself hasn't been trained yet. The roadmap below shows what
> exists today. Weights will be published on GitHub Releases once trained; they are never
> committed to this repository.

## Why from scratch?

Most small open models are trained with large frameworks that hide the
interesting parts. Aupus does every step itself, so each one can be read,
tested and changed:

| Part | What we write ourselves |
|---|---|
| Tokenizer | Byte-level BPE: training (incremental pair counts), GPT-style regex pre-tokenization, lossless encode/decode, special tokens, our own JSON format |
| Data | Streaming download, Unicode normalisation, quality filters, exact + MinHash near-dedup, `uint16` token shards, deterministic resumable dataloader |
| Model | Llama-style decoder: RMSNorm, RoPE, grouped-query attention, SwiGLU, tied embeddings, KV cache |
| Training | AdamW (no decay on norms/embeddings), warmup + cosine schedule, bf16/fp16 autocast, gradient accumulation, `torch.compile` with eager fallback, bit-exact checkpoint resume, Ctrl-C-safe stops, TensorBoard |
| Fine-tuning | Chat template + supervised fine-tuning on human-written conversations, loss on assistant tokens only |
| Evaluation | Validation perplexity and our own log-likelihood harness for HellaSwag, ARC-Easy and PIQA |

**Rules we hold ourselves to:** only human-written, openly licensed training data
(no LLM-generated instruction sets); no `transformers` / `peft` / `trl` /
`accelerate` / `tokenizers` / nanoGPT / LitGPT. Dependencies are PyTorch, NumPy,
PyYAML, safetensors (serialization only), tqdm, TensorBoard and, for downloading
raw text only, Hugging Face `datasets`.

## The model

The planned Aupus 0.1 shape is the `small` preset (`configs/model/small.yaml`):

| | |
|---|---|
| Parameters | 125.3M (100.7M non-embedding) |
| Layers / width | 16 × 768 |
| Attention | 12 query heads, 4 key/value heads (GQA), RoPE, context 1024 |
| MLP | SwiGLU, hidden 2048 |
| Vocabulary | 32,000 byte-level BPE tokens |
| Training data (planned) | ~2.5B tokens of [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (ODC-By 1.0) |
| Fine-tuning data (planned) | OpenAssistant oasst1/oasst2 (Apache-2.0), Databricks Dolly 15k (CC BY-SA 3.0), a small hand-written identity set |

**Honest expectations:** a 125M model trained on a single GPU writes fluent
English and answers simple questions, but it **will** hallucinate facts and
can't reason like large commercial models. It is a learning and research
artifact, not an assistant to rely on.

The full plan, hardware analysis and measured throughput are in
[`docs/PLAN.md`](docs/PLAN.md).

## Roadmap

| # | Milestone | Status |
|---|---|---|
| 1 | Hardware check, throughput benchmark, plan | ✅ done |
| 2 | Project skeleton, configs, tests, CI, license | ✅ done |
| 3 | Byte-level BPE tokenizer | ✅ done |
| 4 | Model architecture (+ overfit test) | ✅ done |
| 5 | Data pipeline on a small sample | ✅ done |
| 6 | Pretraining loop + end-to-end `tiny` run | ✅ done |
| 7 | Full data preparation | ⏳ next |
| 8 | Aupus 0.1 Base pretraining | |
| 9 | Evaluation harness + results | |
| 10 | Supervised fine-tuning → Aupus 0.1 Instruct | |
| 11 | `generate` / `chat` CLI, release export | |
| 12 | Model card, data card, contributing guide | |

## Getting started (development)

Requirements: Python 3.11+ on Linux, macOS or Windows. A GPU isn't needed for development
or tests; everything runs on CPU with the tiny `debug` preset.

```bash
git clone <repo-url> && cd Aupus-AI
python -m venv .venv && source .venv/bin/activate

# 1. Install PyTorch for your machine first:
pip install torch --index-url https://download.pytorch.org/whl/cpu      # CPU only
# pip install torch --index-url https://download.pytorch.org/whl/cu126  # NVIDIA GPU (pick your CUDA version)

# 2. Install Aupus in editable mode with dev tools
pip install -e ".[dev]"
```

Useful commands:

```bash
pytest                                   # test suite
ruff check . && ruff format --check .    # lint + format check
python scripts/check_env.py              # what hardware does this machine have?
python scripts/model_info.py             # exact params and FLOPs/token for every preset
python scripts/benchmark_throughput.py --presets tiny --micro-batch 4   # training tokens/sec (CPU or CUDA)
```

### Tokenizer

The trained 32k tokenizer is committed at `data/tokenizer/aupus-bpe-32k.json`
(with a readable `aupus-bpe-32k.vocab.txt` and measured `aupus-bpe-32k.stats.json`).

```python
from aupus.tokenizer import EOS, Tokenizer

tok = Tokenizer.load("data/tokenizer/aupus-bpe-32k.json")
ids = tok.encode("Hello, world!")  # special-token strings in text stay plain text
tok.decode(ids)  # 'Hello, world!'
tok.special_id(EOS)  # 31994
```

To reproduce it (downloads ~310 MB of FineWeb-Edu text):

```bash
pip install -e ".[data]"
python scripts/download_tokenizer_sample.py   # data/tokenizer_sample/{train,heldout}.jsonl
python scripts/train_tokenizer.py             # data/tokenizer/aupus-bpe-32k.{json,vocab.txt,stats.json}
```

### Data

Normalise → filter → dedup (exact + MinHash) → split → tokenize → `uint16` shards.
See [`docs/DATA.md`](docs/DATA.md) for every rule and the measured results.

```bash
python scripts/prepare_data.py --input data/tokenizer_sample/train.jsonl --out data/pretrain/sample
```

### Training

One YAML file per run in `configs/train/` (strict: a typo in a key is an error). Any key can
be overridden on the command line, and dotted keys reach nested settings:

```bash
python scripts/train.py --config configs/train/tiny.yaml                     # new run -> runs/tiny/
python scripts/train.py --config configs/train/tiny.yaml max_steps=200 optim.lr=2e-3 run_name=tiny-lr2e3
python scripts/train.py --resume runs/tiny                                   # continue after Ctrl-C or a crash
tensorboard --logdir runs                                                    # loss, LR, grad norm, tok/s, samples
```

A run writes everything to `runs/<run_name>/`: the resolved `config.yaml` and `model.yaml`,
`train.log`, `metrics.jsonl`, TensorBoard events, the newest checkpoints (`step_NNNNNN.pt`)
and `best.pt` (lowest validation loss). Press Ctrl-C once to finish the current step, save and
exit. Resuming continues bit for bit where the run stopped, with the same data order and no
repeated or skipped batches.

`precision: auto` uses bf16 on Ampere and newer GPUs and fp32 everywhere else. `compile: true`
uses `torch.compile` where it works and falls back to eager mode where it doesn't (e.g. Windows,
which has no Triton).

**Milestone 6 check run** (`configs/train/tiny.yaml`, 34M params, 19.7M tokens, ~44 min on a
GTX 1650 Super): validation loss fell from 10.48 (a uniform guess over 32k tokens) to 4.95
(perplexity 141), and a hard kill + `--resume` reproduced the original losses exactly. Samples
for the prompt "In 1492, Christopher Columbus":

| Step | Continuation |
|---|---|
| 0 | `overnight overnight250Dem placebo clips legal Linda…` |
| 200 | `is an overview. The results were shown that it was a number of all time…` |
| 1200 | `was an officer. He was a professor of medicine, a professor and a biologist from the University of Virginia…` |

Fluent-looking and factually meaningless, as expected for that size. Details are in
[`docs/PLAN.md`](docs/PLAN.md) §15.

Chat commands will be documented here once Milestone 11 lands.

## Repository layout

```
aupus/          the Python package
  config.py     strict YAML <-> dataclass configs (typos in keys fail loudly)
  tokenizer/    byte-level BPE: pre-tokenizer, trainer, encode/decode
  data/         normalise, filters, dedup, shards, resumable dataloader
  model/        config + size/FLOP accounting, layers, transformer, KV-cache generation
  train/        pretraining loop, AdamW + LR schedule, checkpoints (SFT: milestone 10)
  eval/         perplexity + benchmarks        (milestone 9)
  cli/          generate / chat                (milestone 11)
configs/model/  model presets: debug (~6.5M), tiny (~34M), small (~125M), base (~348M)
configs/data/   data-pipeline settings (filters, dedup, split, shard size)
configs/train/  training-run configs: tiny (M6 check), small (Aupus 0.1 Base, M8)
scripts/        entry points
tests/          pytest suite (runs on CPU in CI)
docs/           PLAN.md (plan, measurements, decisions), DATA.md (sources, pipeline)
```

Data, checkpoints, logs and release weights stay out of git (see `.gitignore`).

## License

[Apache-2.0](LICENSE). Each training dataset keeps its own license; they are
documented in [`docs/DATA.md`](docs/DATA.md) (and later the model card).
