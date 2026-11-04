# Aupus 0.1 — Phase 0 Plan

Status: **approved** (2026-11-04). Decisions are recorded in §10.
Date: 2026-11-04

## 1. Detected hardware (from `scripts/check_env.py`)

| Item | Value |
|---|---|
| OS | Ubuntu 24.04.4 LTS, kernel 7.0.0 |
| Python | 3.12.3 (system); project venv at `.venv/` |
| PyTorch | 2.15.0+cpu (CPU-only build, installed from the PyTorch CPU index) |
| CPU | AMD Ryzen 7 7730U (Zen 3 laptop chip), 8 cores / 16 threads, AVX2 + FMA, **no AVX-512 / no BF16 instructions** |
| GPU | AMD Radeon integrated graphics (Barcelo, `gfx90c`), **0.5 GiB dedicated VRAM**, shares system RAM (7.5 GiB GTT) |
| CUDA / ROCm usable by PyTorch | **None** |
| bf16 | Works on CPU but is ~10× slower than fp32 here (no native BF16 units), so it is **not usable** |
| RAM | 15 GiB total (~11.7 GiB available) |
| Disk (project drive) | 274 GiB free |

**Verdict: this machine cannot train Aupus 0.1 at the planned scale.** It is fine
for development: tokenizer, data pipeline, model code, tests, CI, and small
`debug`/`tiny` runs. A ~125M-parameter model needs a rented GPU (see §3).

Why not use the integrated GPU? ROCm doesn't officially support `gfx90c`. It
only works with the `HSA_OVERRIDE_GFX_VERSION` hack, and the ROCm PyTorch
wheels are a multi-GB download. Even when it works, the GPU has about 2 TFLOPS
fp32 and shares the CPU's DDR4 bandwidth. That's at most a 2–4× speedup, so
`small` would still take weeks. It's not worth the instability.

## 2. Measured throughput (`scripts/benchmark_throughput.py`)

Each step is a full training step (forward + backward + grad clipping + AdamW)
with context 1024, fp32, 8 threads, and random tokens. The model is a compact
copy of the planned architecture (RMSNorm, RoPE, GQA with SDPA, SwiGLU, tied
embeddings).

| Preset | Shape (d / layers / heads / kv / ffn / vocab) | Params (total / non-emb.) | Micro-batch | Tokens/s | Achieved GFLOP/s | Peak RAM |
|---|---|---|---|---|---|---|
| `debug` | 256 / 6 / 4 / 2 / 704 / 8192 | 6.5M / 4.4M | 1 | 3,127 | 152 | 0.7 GiB |
| `tiny` | 512 / 6 / 8 / 4 / 1408 / 32000 | 34.1M / 17.7M | 4 | 1,170 | 261 | 3.5 GiB |
| `tiny` + `torch.compile` | same | same | 4 | **1,406** | 314 | — |
| `small` | 768 / 16 / 12 / 4 / 2048 / 32000 | **125.3M** / 100.7M | 1 | 303 | 251 | 3.4 GiB |
| `base` | 1024 / 28 / 16 / 4 / 2816 / 32000 | **348.4M** / 315.7M | 1 | 117 | 265 | 7.9 GiB |

Other findings:
- 8 threads (one per physical core) beat 16 (1,170 vs 1,036 tok/s).
- `torch.compile` works on CPU (gcc is present) and gives about 1.2×.
- CPU bf16 autocast is unusably slow on this CPU (aborted after more than 9 minutes for 3 steps).
- About 250–315 GFLOP/s achieved, against a theoretical fp32 peak of roughly 770 GFLOP/s (8 cores × 32 FLOP/cycle × ~3 GHz). That's 35–40% utilisation, which is normal for CPU.
- These numbers come from short bursts. A laptop running for days will throttle, so real numbers will be worse.

### Real CPU training-time estimates at ~20 tokens/parameter

| Preset | Token budget | Time on this laptop (best case, compiled) |
|---|---|---|
| `debug` | 130M | ~10 h |
| `tiny` | 680M | **~5.6 days** non-stop |
| `small` | 2.5B | **~80 days** |
| `base` | 7.0B | **~1.6 years** |

## 3. Options for training Aupus 0.1 Base

The cloud numbers below are **calculated, not measured**. They assume about 40%
utilisation with bf16 + `torch.compile` + flash attention. The first thing done
on any rented GPU is to re-run the benchmark (extended with a CUDA path) and
replace these numbers. Prices are rough marketplace ranges (RunPod, vast.ai,
Lambda, etc.). **Check current prices before renting.**

Training compute: `small` ≈ 0.83 GFLOP/token × 2.5B tokens ≈ **2.1e18 FLOP**;
`base` ≈ 2.27 GFLOP/token × 7.0B tokens ≈ **1.6e19 FLOP**.

| Option | `small` (125M, 2.5B tok) | `base` (350M, 7B tok) | Notes |
|---|---|---|---|
| **A. Rent 1× RTX 4090 (24 GB)**, ~$0.3–0.7/h | ~9 h → **~$3–6** | ~66 h → ~$20–46 | Cheapest per FLOP; bf16 supported |
| B. Rent 1× A100 80 GB, ~$1.0–1.8/h | ~5 h → ~$5–8 | ~35 h → ~$35–63 | Most predictable |
| C. Rent 1× H100, ~$2–3/h | ~2–3 h → ~$5–8 | ~15 h → ~$30–45 | Fastest; small models under-use it |
| D. Free Kaggle/Colab T4 (16 GB, fp16 only) | ~25–30 h across many sessions | impractical | Free but slow. Needs fp16 + GradScaler and lots of resumes (our checkpointing supports this) |
| E. Laptop CPU only | ~80 days | — | Not realistic |
| F. Laptop CPU, `tiny` only | ~5.6 days | — | Possible, but quality would be very poor (mostly fluent-ish text, little QA ability) |

Add about 1.5–2× on top for SFT, evaluation, a restarted run and idle time. A
realistic total cloud budget for `small` is about **$10–25**.

### Recommendation

- **Model: `small` (~125M) for Aupus 0.1.** It matches the spec's target, is
  cheap to train on rented hardware, and the measured shapes/FLOPs are known.
  `base` (350M) is affordable (~$40–80 total) and noticeably better, but it
  costs 7–8× more and only makes sense once the pipeline is proven on `small`.
- **Hardware: develop locally, train on a rented single GPU (Option A or B).**
  Milestones 2–6 (code, tests, `tiny` end-to-end run) run on this laptop.
  Milestones 7–10 (full data, pretraining, eval, SFT) run on the rented GPU.
  Milestone 7 (data prep) can run on either machine; I'll recommend which once
  the tokenizer's real encode speed is measured in Milestone 3.

## 4. Chosen configuration for Aupus 0.1

| Setting | Value | Why |
|---|---|---|
| Architecture | `small`: d_model 768, 16 layers, 12 query heads, 4 KV heads (head_dim 64), SwiGLU hidden 2048, RMSNorm, RoPE (θ=10000), tied embeddings, no biases | ~125M params. A deeper/narrower shape (16 layers instead of 12) works better at this scale (MobileLLM). GQA 3:1 makes the KV cache 3× smaller at no real quality cost |
| Vocab | 32,000 byte-level BPE + 7 special tokens | Spec default. At 125M, embeddings are 20% of params, which is acceptable |
| Context length | 1024 | Spec default. QA/chat at this size rarely benefits from more |
| Token budget | **2.5B tokens** (20 × 125M) | Chinchilla-optimal. Optional: training past Chinchilla (e.g. 5B tokens) clearly helps small models and costs only ~2× more. Decide after the first run |
| Effective batch | **262,144 tokens/step** (256 sequences × 1024) → ~9,550 steps | Middle ground: small enough to give enough optimizer steps, large enough for stable gradients |
| Micro-batch × grad-accum | Set from the GPU benchmark, e.g. 16 × 16 (24 GB) or 32 × 8 (80 GB) | Measured on the rented GPU, not guessed |
| Optimizer | AdamW, β=(0.9, 0.95), eps 1e-8, weight decay 0.1 (none on norms/embeddings), peak LR 6e-4, 2% linear warmup (~190 steps), cosine decay to 6e-5, grad clip 1.0 | GPT-3-small / Llama-style defaults known to work at this scale |
| Precision | bf16 autocast on GPU. fp16 + GradScaler fallback (T4). fp32 on CPU | Section 2 shows CPU bf16 is too slow on this CPU |
| `torch.compile` | On by default, `--no-compile` flag | 1.2× on CPU; usually 1.3–2× on GPU |
| Checkpoints | Every 500 steps, keep last 3 + best-val | ~1.5 GB each (fp32 weights + AdamW state) |
| `tiny` (M6 e2e) | CPU, context 512, effective batch 16K tokens, ~25 min ≈ 2M tokens | Checks the pipeline end to end (loss falls, samples improve). It won't produce a usable model |
| `debug` | vocab 8192, ~6.5M params | Unit tests, overfit-one-batch, CI on CPU |

**SFT (Aupus 0.1 Instruct):** start from the best-val Base checkpoint. Peak LR
about 3e-5 (≈ 1/20 of pretraining), 2–3 epochs over ~20–30K conversations,
loss only on assistant tokens, early-stop on val loss. A few minutes to an hour
on the rented GPU.

## 5. Data plan (to be documented in `docs/DATA.md`)

| Source | Use | License | Size used |
|---|---|---|---|
| FineWeb-Edu (`HuggingFaceFW/fineweb-edu`, `sample-10BT`) | Pretraining | ODC-By 1.0 | ~2.7B raw tokens → ~2.5B after our filters + dedup + 0.5% val split |
| ~~English Wikipedia~~ (**declined for 0.1**) | — | CC BY-SA 4.0 / GFDL | Not used: share-alike licensing of training data is a grey area for weight licensing. May be revisited for a later version |
| OpenAssistant oasst1 + oasst2, English, top-ranked branches | SFT | Apache-2.0 | ~10–15K conversations |
| Databricks Dolly 15k | SFT | CC BY-SA 3.0 | ~15K examples |
| `data/identity.jsonl` (hand-written by us, reviewed by you) | SFT | Apache-2.0 (ours) | 50–100 examples |

Pipeline: streaming download → NFC normalisation → quality filters → exact
(hash) + MinHash/LSH near-dedup (our own code, ~1.4 GB RAM for ~2.7M docs ×
128 permutations) → 0.5% held-out val → `uint16` memmap shards with `<|eos|>`
separators.

## 6. Disk, RAM and VRAM estimates

| Item | Disk |
|---|---|
| `.venv` (CPU torch) | ~1 GB (already used) |
| FineWeb-Edu download (streamed, ~2.7B tokens of parquet) | ~8–10 GB, deletable after processing |
| Cleaned text intermediate | ~10–12 GB, deletable |
| Token shards (`uint16`, 2 bytes/token) | ~5 GB (`small`), ~14 GB (`base`) |
| Checkpoints (`small`: 3 + best) | ~6 GB |
| Tokenizer training sample | ~0.3–0.5 GB |
| **Peak total** | **~30 GB** (274 GB free: OK) |

- RAM: data prep fits in 15 GB if dedup works on shards. Training loads shards with memmap, so RAM use is small.
- VRAM (`small`, bf16, micro-batch 16 × 1024): estimated ~10–14 GB (weights + grads + AdamW ≈ 2 GB, activations + logits ≈ 8–12 GB). Fits a 24 GB card. This will be measured on the GPU.

## 7. Milestones & compute time

| # | Milestone | Where | Compute time |
|---|---|---|---|
| 1 | Phase 0: specs, benchmark, this plan | laptop | done |
| 2 | Skeleton, `pyproject.toml`, configs, tests, CI, LICENSE | laptop | — |
| 3 | Tokenizer + tests (train on ~300 MB sample) | laptop | ~0.3 GB download; BPE training ~10–40 min (measured then) |
| 4 | Model + tests (overfit `debug`) | laptop | minutes |
| 5 | Data pipeline on a small sample | laptop | minutes |
| 6 | Pretraining loop + `tiny` e2e run | laptop | ~25 min |
| 7 | Full data prep (**ask first**) | laptop or GPU box | ~1–3 h, ~10 GB download |
| 8 | Aupus 0.1 Base pretraining (**ask first**) | rented GPU | ~5–9 h |
| 9 | Eval harness (HellaSwag, ARC-Easy, PIQA, val ppl) | rented GPU or laptop | <1 h |
| 10 | SFT → Aupus 0.1 Instruct | rented GPU | <1 h |
| 11 | generate/chat CLI, release export | laptop | — (CPU inference of 125M is fast enough) |
| 12 | README, MODEL_CARD, DATA, CONTRIBUTING | laptop | — |

## 8. Honest quality expectations

A 125M model trained on 2.5B tokens is roughly comparable to GPT-2 small
(124M, trained on far more tokens of less-filtered data). It may come out
slightly better on reasoning-flavoured benchmarks thanks to FineWeb-Edu.
Expect:
- fluent, mostly grammatical English and simple factual answers;
- frequent hallucinations, weak multi-step reasoning, short coherent span;
- HellaSwag ~30% (random 25%), ARC-Easy ~40–45% (random 25%), PIQA ~60–63% (random 50%).

These are ballpark figures based on published results for similarly sized models, not promises.

## 9. Risks

| Risk | Mitigation |
|---|---|
| Cloud instance is pre-empted or dies mid-run | Full resume (model/optimizer/scheduler/RNG/dataloader). Copy checkpoints off the box regularly |
| GPU throughput lower than calculated | Re-benchmark first on the rented GPU. Re-estimate before committing hours |
| Pure-Python BPE encode too slow for 2.5B tokens | Multiprocessing (16 threads) + caching repeated pre-token chunks. If still too slow, a small C extension (allowed by spec) |
| 15 GB RAM limits dedup / tokenizer training | Shard-wise processing, NumPy arrays, LSH bands on disk if needed |
| Laptop thermal throttling during long CPU jobs | Keep laptop runs short (M6 ≤ 30 min) |
| Wikipedia CC BY-SA licence ambiguity | Avoided: Wikipedia not used in 0.1 |
| Small SFT set → overfitting | Low LR, few epochs, val-loss early stopping |
| PyTorch API drift (2.15 installed) | Pin versions in `pyproject.toml` / `uv.lock` |

## 10. Decisions

| # | Question | Decision |
|---|---|---|
| 1 | Where to train Base (M7–M10) | **Decide before Milestone 7.** Milestones 2–6 run on the laptop |
| 2 | YAML parser | **`pyyaml` allowed** (the one dependency added beyond the spec's list). Our loader adds YAML 1.2 float parsing so `lr: 3e-4` is a float, not a string |
| 3 | `uv` | Pending. Until then a plain `python -m venv .venv` + `pip install -e ".[dev]"` is used; `pyproject.toml` is uv-compatible, so switching later only adds a `uv.lock` |
| 4 | Wikipedia slice | **No.** FineWeb-Edu only for 0.1 (simplest licensing) |
| 5 | Project root | This repo's root (`Aupus-AI/`) is the spec's `aupus-ai/` |
| 6 | CI | GitHub Actions installs the CPU PyTorch wheel from the PyTorch CPU index |
| 7 | Build backend | `setuptools` (standard packaging; not a runtime dependency) |
