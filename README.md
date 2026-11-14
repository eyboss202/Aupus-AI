<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/wordmark-dark.svg">
    <img src="docs/assets/wordmark-light.svg" height="80" alt="vulpes">
  </picture>
</p>

<p align="center">
  Small language models built completely from scratch:<br>
  our own tokenizer, model, training, fine-tuning and chat, in plain PyTorch.
</p>

<p align="center">
  <a href="https://github.com/eyboss202/Aupus-AI/actions/workflows/ci.yml"><img src="https://github.com/eyboss202/Aupus-AI/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-1d1c22" alt="License: Apache-2.0"></a>
  <img src="https://img.shields.io/badge/python-3.11%20%7C%203.12-1d1c22" alt="Python 3.11 | 3.12">
  <img src="https://img.shields.io/badge/PyTorch-plain-1d1c22" alt="Plain PyTorch">
  <img src="https://img.shields.io/badge/Vulpes%200.1-trained-B5471F" alt="Vulpes 0.1: trained">
  <img src="https://img.shields.io/badge/Vulpes%20Flos%201.0-in%20preparation-E57952" alt="Vulpes Flos 1.0: in preparation">
</p>

<p align="center">
  <a href="#the-vulpes-family">Models</a> ·
  <a href="#talk-to-vulpes">Chat</a> ·
  <a href="docs/TRAINING.md">Train it yourself</a> ·
  <a href="MODEL_CARD.md">Model card</a> ·
  <a href="docs/DATA.md">Data</a> ·
  <a href="docs/PLAN-FLOS.md">Flos plan</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/web-chat-dark.png">
    <img src="docs/assets/web-chat-light.png" width="760" alt="The Vulpes browser chat: the orange fox, &quot;What shall we learn today?&quot;, a message box and four suggested questions">
  </picture>
</p>

---

**Vulpes** is a family of small English language models made **from nothing**: the tokenizer,
the transformer, the training loop, fine-tuning, evaluation and the chat are all written in
this repository. There are no pretrained weights, no other model's outputs, no external model
APIs and no high-level model libraries.

It is also a learning project. The code is meant to be read: docstrings explain *why* each
piece exists, not only what it does, and every plan, decision and measurement is written down
in [`docs/`](docs/).

## The Vulpes family

*Vulpes* is Latin for fox, the animal in the logo. Each model takes the previous one further:

| Model | Size | What it is | Status |
|---|---|---|---|
| **Vulpes 0.1** | 125M | The first: English text and simple questions, 2.5B tokens, one rented GPU for ~4 h | ✅ **trained** (2026-11-10), Base and Instruct |
| **Vulpes Flos 1.0** | 348M | Vulpes 0.1 taken as far as free Kaggle GPUs allow: 2.8× the size, 4–6× the text, twice the memory | 🚧 **in preparation** ([plan](docs/PLAN-FLOS.md)) |
| **Vulpes Lusitanus 1.0** | ~1B | Code, Portuguese and tools, on a new tokenizer | 📝 planned, after Flos ([PDF, Portuguese](docs/Vulpes-Lusitanus-1.0-Plano.pdf)) |

*Flos* is Latin for flower; *Lusitanus*, for Portuguese. Until November 2026 the project was
called Aupus, and the repository still is. Tokenizers, datasets, checkpoints and releases made
under the old name load as they are.

## Vulpes 0.1, measured

Zero-shot, on the full test sets, scored the way lm-evaluation-harness scores them
(`scripts/evaluate.py`; raw numbers in `runs/vulpes-0.1-*/eval_best.json`):

| | Vulpes 0.1 Base | Vulpes 0.1 Instruct | Random guess |
|---|---|---|---|
| HellaSwag (acc_norm, 10,042 questions) | **31.3%** | 31.2% | 25% |
| ARC-Easy (acc, 2,376) | **52.4%** | 52.2% | 25% |
| PIQA (acc_norm, 1,838) | **61.9%** | 61.0% | 50% |
| FineWeb-Edu validation perplexity (14.6M tokens) | **22.9** | 24.8 | 32,000 |
| Bits per byte | 0.983 | 1.007 | — |

That is roughly where GPT-2 small (124M) lands on HellaSwag and PIQA, and well ahead of it on
ARC-Easy, the gain FineWeb-Edu's educational text is known for. What it writes, unedited
(Instruct, from the fine-tuning log):

> **User:** Why is the sky blue?<br>
> **Vulpes:** The sky is blue because the light from the sun moves around the planet. Many
> people experience blue light because sunlight is scattered by more than 90 percent of the
> light from the sun. That is why you see blue when the sun moves around the sky.

Fluent, on topic, half right. **Vulpes 0.1 makes things up, can't reason in several steps and
speaks only English.** It is a research and learning artifact, not an assistant to rely on;
[`MODEL_CARD.md`](MODEL_CARD.md) has the details. The 0.1 weights were fine-tuned before the
rename, so they still introduce themselves as Aupus: [`data/identity.jsonl`](data/identity.jsonl)
now says Vulpes, and Flos is the first model to learn it.

| | How it was made |
|---|---|
| Tokenizer | 32,000-id byte-level BPE trained here on 300 MB of FineWeb-Edu in 47 s; 4.59 bytes per token, lossless |
| Data | The first 4 of 14 FineWeb-Edu `sample-10BT` files, filtered and deduplicated here: **2.94B tokens** |
| Model | Llama-style decoder, 16 layers × 768, 12 query / 4 key-value heads, SwiGLU, RoPE, context 1,024, tied embeddings |
| Pretraining | 2.5B tokens, 9,537 steps of 262,144 tokens, on one rented A100 80 GB: **~4 h at ~167,000 tokens/s** (bf16, `torch.compile`, 138 TFLOPS) |
| Fine-tuning | oasst2 + Dolly 15k + a hand-written identity set, loss on replies only; best at step 600 of 1,808, stopped early at 900 |

## Vulpes Flos 1.0, next

The same recipe, scaled to what Kaggle's free GPUs can do in a few months, at **zero cost**:

| | Vulpes 0.1 | Vulpes Flos 1.0 |
|---|---|---|
| Parameters | 125M | **348M** (28 layers × 1,024, the `base` preset) |
| Context | 1,024 | **2,048** tokens |
| Pretraining text | 2.5B tokens | **10–15B+** tokens, the whole FineWeb-Edu `sample-10BT` |
| Hardware | one rented A100, ~4 h | Kaggle's free 2× T4, in chained 12-hour sessions (and its TPUs, if a trial pays off) |
| Expected HellaSwag / ARC-Easy | 31% / 52% (measured) | 36–41% / 56–62% (estimates) |

New in the training code for it: two-GPU data parallelism (DDP), a warmup–stable–decay schedule
whose length can change mid-run, fp16 with loss scaling on GPUs without bf16, and a session
cycle that resumes the run bit for bit every 12 hours. Milestones K1–K10, decisions and risks:
[`docs/PLAN-FLOS.md`](docs/PLAN-FLOS.md) (summary of the full
[plan in Portuguese](docs/Vulpes-Flos-1.0-Plano.pdf)).

## Why from scratch?

Most small open models are trained with large frameworks that hide the interesting parts.
Vulpes does every step itself, so each one can be read, tested and changed:

| Part | What we write ourselves |
|---|---|
| Tokenizer | Byte-level BPE: training (incremental pair counts), GPT-style regex pre-tokenization, lossless encode/decode, special tokens, our own JSON format |
| Data | Download with checksums, Unicode normalisation, Gopher quality filters, exact + MinHash near-dedup, `uint16` token shards, a deterministic resumable dataloader |
| Model | Llama-style decoder: RMSNorm, RoPE, grouped-query attention, SwiGLU, tied embeddings, KV cache |
| Training | AdamW (no decay on norms or embeddings), cosine and warmup–stable–decay schedules, bf16/fp16 autocast, gradient accumulation, multi-GPU DDP, `torch.compile` with eager fallback, bit-exact resume, safe stops, TensorBoard |
| Fine-tuning | Chat template and supervised fine-tuning on human-written conversations, loss on the assistant's replies only |
| Evaluation | Validation perplexity and our own log-likelihood harness for HellaSwag, ARC-Easy and PIQA |
| Inference | Streaming generation with a KV cache, a terminal chat, a browser chat and a safetensors release format |

**Rules we hold ourselves to:** only human-written, openly licensed training data (no
LLM-generated instruction sets); no `transformers` / `peft` / `trl` / `accelerate` /
`tokenizers` / nanoGPT / LitGPT. Dependencies are PyTorch, NumPy, PyYAML, safetensors
(serialization only), tqdm, TensorBoard and, for downloading raw text only, Hugging Face
`datasets`.

## Talk to Vulpes

The weights aren't published yet (they will go to GitHub Releases, never into the repository).
With a release folder, yours from [`docs/TRAINING.md`](docs/TRAINING.md) or a published one:

```bash
python scripts/chat_web.py --model release/vulpes-0.1-instruct     # in the browser: http://127.0.0.1:8000
python scripts/chat.py --model release/vulpes-0.1-instruct         # in the terminal
bash vulpes.sh                                                     # one command: sets up Python, PyTorch and Vulpes, opens the browser chat
```

**In the browser**, replies stream as they are written, with Stop, "Try again", "Edit", Copy,
"New chat" (with Undo), the sampling settings and a light or dark theme. A meter shows how much
of the model's memory the conversation fills, and a picker switches between the models found in
`release/` and `runs/` without losing the conversation. "Save chat" writes the conversation to
`data/chats/` (never committed) in the fine-tuning format, to correct by hand and train on. It
runs on your computer only: it listens on 127.0.0.1 and refuses requests from other sites.

**In the terminal**, the same in text: the fox in orange in a welcome panel, replies that stream
in word-wrapped with their length and speed, and commands: `/new`, `/retry`, `/save`, `/info`,
`/model` (list and switch models), `/settings` (temperature, length, top-p, top-k, repetition
penalty) and `/exit`. `NO_COLOR=1` turns the colours off; `VULPES_THEME=light` suits a light
terminal.

Neither ever changes the model by itself: Vulpes has no internet access and no memory between
conversations.

## Getting started

Requirements: Python 3.11+ on Linux, macOS or Windows. A GPU isn't needed for development or
tests; everything runs on CPU with the tiny `debug` preset.

```bash
git clone https://github.com/eyboss202/Aupus-AI.git && cd Aupus-AI
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate

# 1. PyTorch for your machine first
pip install torch --index-url https://download.pytorch.org/whl/cpu        # CPU only
# pip install torch --index-url https://download.pytorch.org/whl/cu126    # NVIDIA GPU (pick your CUDA version)

# 2. Vulpes, editable, with the dev tools (add ,data for the download scripts)
pip install -e ".[dev]"
```

```bash
pytest                                   # the test suite (CPU, a couple of minutes)
ruff check . && ruff format --check .    # lint and format
python scripts/check_env.py              # what can this machine train?
python scripts/model_info.py             # exact parameters and FLOPs per token of every preset
```

## The whole pipeline

Each stage is one script, and each has its own guide in [`docs/`](docs/).

```bash
# 1. Tokenizer (committed: data/tokenizer/vulpes-bpe-32k.json; this rebuilds it exactly)
python scripts/download_tokenizer_sample.py && python scripts/train_tokenizer.py

# 2. Pretraining data: download (SHA-256 checked, resumable), then filter, dedup and tokenize
python scripts/download_fineweb_edu.py --files 4          # Vulpes 0.1: 8.6 GB; Flos uses all 14 files
python scripts/prepare_data.py --source parquet --input data/raw/fineweb_edu --out data/pretrain/fineweb_edu

# 3. Pretraining (Ctrl-C stops safely; --resume continues bit for bit)
python scripts/train.py --config configs/train/small.yaml
python scripts/train.py --resume runs/vulpes-0.1-base
python scripts/train_status.py --watch 30                 # progress, speed, time left, GPU heat

# 4. Evaluation
python scripts/download_eval_data.py
python scripts/evaluate.py --model runs/vulpes-0.1-base/best.pt --val-data data/pretrain/fineweb_edu

# 5. Fine-tuning: Base -> Instruct
python scripts/prepare_sft_data.py
python scripts/sft.py --config configs/sft/vulpes-0.1-instruct.yaml

# 6. Release: weights-only safetensors, model config, tokenizer, checksums, license, model card
python scripts/export_release.py --checkpoint runs/vulpes-0.1-instruct/best.pt --out release/vulpes-0.1-instruct
```

Where to train:

| Where | Guide | For |
|---|---|---|
| Your own NVIDIA GPU | [`docs/TRAINING.md`](docs/TRAINING.md) | Vulpes 0.1 on 4 GB: ~16–17 days, stopped and resumed as often as you like |
| A rented GPU on RunPod | [`docs/RUNPOD.md`](docs/RUNPOD.md) | Vulpes 0.1 in hours, for a few dollars: set up, trained, copied home and stopped by `scripts/runpod/` |
| Kaggle's free GPUs | [`docs/KAGGLE.md`](docs/KAGGLE.md) | Vulpes Flos 1.0 on 2× T4, one 12-hour session at a time, by `scripts/kaggle/` |

Every run is one strict YAML file in `configs/train/` (a typo in a key is an error), and any key
can be overridden on the command line (`optim.lr=3e-4`). A run writes everything to
`runs/<run_name>/`: the resolved config, `train.log`, `metrics.jsonl`, TensorBoard events, the
newest checkpoints and `best.pt`. Resuming continues with the same data order, the same random
state and no repeated or skipped batch, on the same machine or another one.

## Repository layout

```
vulpes/           the Python package
  config.py       strict YAML <-> dataclass configs
  tokenizer/      byte-level BPE: pre-tokenizer, trainer, encode/decode, chat template
  data/           normalise, filters, dedup, shards, resumable dataloader, fine-tuning data
  model/          config and size/FLOP accounting, layers, transformer, KV-cache generation, loading
  train/          pretraining loop, DDP, fine-tuning, AdamW and LR schedules, checkpoints
  eval/           perplexity, bits per byte, HellaSwag / ARC-Easy / PIQA
  cli/            terminal chat, browser chat (one HTML page), streaming, fonts and logo
configs/model/    presets: debug (~6.5M), tiny (~34M), small (125M, Vulpes 0.1), base (348M, Vulpes Flos 1.0)
configs/data/     data-pipeline settings
configs/train/    pretraining runs: tiny (check run), small (Vulpes 0.1), flos-kaggle (Vulpes Flos 1.0), ...
configs/sft/      fine-tuning runs
scripts/          entry points; runpod/ (rented GPU) and kaggle/ (free GPUs) automate whole runs
tests/            pytest suite, CPU only, run by CI
docs/             plans (PLAN.md for 0.1, PLAN-FLOS.md), guides (TRAINING, RUNPOD, KAGGLE, IDENTITY), DATA.md
  assets/         the fox: logo.svg, logo-small.svg (16–32 px), wordmarks, screenshots
vulpes.sh         one-command launcher for the chat
MODEL_CARD.md     what the models are, how they were made, their limits
```

Data, checkpoints, logs and release weights stay out of git (see `.gitignore`).

## The fox

The logo is the author's own drawing ([`logoVulpes.png`](logoVulpes.png)), vectorised into
[`docs/assets/logo.svg`](docs/assets/logo.svg) and a heavier
[`logo-small.svg`](docs/assets/logo-small.svg) for tab icons and small sizes. The colour is fox
orange `#E57952`, deepened to `#B5471F` where it is text on paper (5:1 contrast). The name is
set in Newsreader, the interface and the model's replies in Instrument Sans; in the terminal the
fox is drawn in quarter-block characters.

## License

[Apache-2.0](LICENSE), for the code and the weights. The chat bundles two typefaces under the
SIL Open Font License 1.1, Newsreader and Instrument Sans (`vulpes/cli/fonts/OFL-*.txt`). Each
training dataset keeps its own license: FineWeb-Edu (ODC-By 1.0), OpenAssistant oasst2
(Apache-2.0) and Dolly 15k (CC BY-SA 3.0); see [`docs/DATA.md`](docs/DATA.md).
