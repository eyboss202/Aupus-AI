# Aupus AI

**Aupus 0.1** is a small English language model that writes text and answers simple
questions, built **completely from scratch**: tokenizer, transformer, training loop,
fine-tuning and inference engine are all written in this repository with plain
PyTorch. There are no pretrained weights, no external model APIs and no high-level
model libraries.

It is also a learning project. The code is meant to be read: docstrings explain
*why* each piece exists, not just what it does.

> **Status: pre-alpha, under construction.** No model has been trained yet. The
> roadmap below shows what exists today. Weights will be published on GitHub
> Releases once trained; they are never committed to this repository.

## Why from scratch?

Most small open models are trained with large frameworks that hide the
interesting parts. Aupus does every step itself, so each one can be read,
tested and changed:

| Part | What we write ourselves |
|---|---|
| Tokenizer | Byte-level BPE: training (incremental pair counts), GPT-style regex pre-tokenization, lossless encode/decode, special tokens, our own JSON format |
| Data | Streaming download, Unicode normalisation, quality filters, exact + MinHash near-dedup, `uint16` token shards, deterministic resumable dataloader |
| Model | Llama-style decoder: RMSNorm, RoPE, grouped-query attention, SwiGLU, tied embeddings, KV cache |
| Training | AdamW, warmup + cosine schedule, bf16 autocast, gradient accumulation, `torch.compile`, full checkpoint resume, TensorBoard |
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
| 3 | Byte-level BPE tokenizer | ⏳ next |
| 4 | Model architecture (+ overfit test) | |
| 5 | Data pipeline on a small sample | |
| 6 | Pretraining loop + end-to-end `tiny` run | |
| 7 | Full data preparation | |
| 8 | Aupus 0.1 Base pretraining | |
| 9 | Evaluation harness + results | |
| 10 | Supervised fine-tuning → Aupus 0.1 Instruct | |
| 11 | `generate` / `chat` CLI, release export | |
| 12 | Model card, data card, contributing guide | |

## Getting started (development)

Requirements: Python 3.11+, Linux or macOS. A GPU isn't needed for development
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
python scripts/benchmark_throughput.py --presets tiny --micro-batch 4   # training tokens/sec
```

Training, data preparation and chat commands will be documented here as their
milestones land.

## Repository layout

```
aupus/          the Python package
  config.py     strict YAML <-> dataclass configs (typos in keys fail loudly)
  tokenizer/    byte-level BPE                 (milestone 3)
  data/         data pipeline                  (milestone 5)
  model/        model config, layers, generation
  train/        pretraining + SFT loops        (milestones 6, 10)
  eval/         perplexity + benchmarks        (milestone 9)
  cli/          generate / chat                (milestone 11)
configs/model/  model presets: debug (~6.5M), tiny (~34M), small (~125M), base (~348M)
configs/train/  training-run configs
scripts/        entry points
tests/          pytest suite (runs on CPU in CI)
docs/           plan, data documentation
```

Data, checkpoints, logs and release weights stay out of git (see `.gitignore`).

## License

[Apache-2.0](LICENSE). Each training dataset keeps its own license; they will be
documented in `docs/DATA.md` and the model card.
