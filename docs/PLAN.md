# Vulpes 0.1 — Phase 0 Plan

Status: **approved** (2026-11-04). Decisions are recorded in §10.
Progress: Milestones 1–6 done (tokenizer §11, model §13, data pipeline §14, pretraining loop §15; all 2026-11-06).
Milestone 7 done on the desktop (2026-11-09): 2.94B train + 14.6M val tokens (§16). Milestone 8 configured
(`small`, 2.5B tokens, decision 39) after a 1-hour pilot (§18), not started yet: the user starts it.
Milestones 9–12: code, tests and docs done (§17), and dry-run on the GPU from the pilot checkpoint, with the
browser chat added (§19); their real runs need the trained Base model.
Development moved to a new machine on 2026-11-06 (§12). Since 2026-11-10 M8 is planned on a rented
RunPod RTX 4090 (decisions 47–49, docs/RUNPOD.md).
Since 2026-11-14 the model is called **Vulpes** (it was Aupus), with a fox in orange as its mark (decision 51);
this document uses the new name throughout.
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

**Verdict: this machine cannot train Vulpes 0.1 at the planned scale.** It is fine
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

## 3. Options for training Vulpes 0.1 Base

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

- **Model: `small` (~125M) for Vulpes 0.1.** It matches the spec's target, is
  cheap to train on rented hardware, and the measured shapes/FLOPs are known.
  `base` (350M) is affordable (~$40–80 total) and noticeably better, but it
  costs 7–8× more and only makes sense once the pipeline is proven on `small`.
- **Hardware: develop locally, train on a rented single GPU (Option A or B).**
  Milestones 2–6 (code, tests, `tiny` end-to-end run) run on this laptop.
  Milestones 7–10 (full data, pretraining, eval, SFT) run on the rented GPU.
  Milestone 7 (data prep) can run on either machine; I'll recommend which once
  the tokenizer's real encode speed is measured in Milestone 3.

## 4. Chosen configuration for Vulpes 0.1

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

**SFT (Vulpes 0.1 Instruct):** start from the best-val Base checkpoint. Peak LR
about 3e-5 (≈ 1/20 of pretraining), 2–3 epochs over ~20–30K conversations,
loss only on assistant tokens, early-stop on val loss. A few minutes to an hour
on the rented GPU.

## 5. Data plan (documented in `docs/DATA.md`)

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
| 2 | Skeleton, `pyproject.toml`, configs, tests, CI, LICENSE | laptop | done |
| 3 | Tokenizer + tests (train on ~300 MB sample) | desktop | done: 310 MB download in ~1 min; BPE training **47 s** (§11) |
| 4 | Model + tests (overfit `debug`) | desktop | done: overfit test ~6 s on CPU; CUDA benchmark (§13) |
| 5 | Data pipeline on a small sample | desktop | done: 300 MB in 49 s → 64.8M tokens (§14) |
| 6 | Pretraining loop + `tiny` e2e run | desktop GPU | done: 1,200 steps / 19.7M tokens in ~42 min (§15) |
| 7 | Full data prep | desktop | done: 8.6 GB download in 25 min + 68 min CPU on 3 workers → 2.94B tokens (§16) |
| 8 | Vulpes 0.1 Base pretraining (user starts it) | desktop GTX 1650 Super (decision 1) | configured: `small`, 2.5B tokens, ~16–17 days of GPU time (§18, decision 39); 1-hour pilot done |
| 9 | Eval harness (HellaSwag, ARC-Easy, PIQA, val ppl) | desktop GPU | code done (§17); dry run: a full eval of `small` takes ~14 min (§19) |
| 10 | SFT → Vulpes 0.1 Instruct | desktop GPU | code + data done; dry run: ~3.5 h for 3 epochs, less with early stopping (§19). Needs `data/identity.jsonl` |
| 11 | generate/chat CLI, browser chat, release export | desktop | done (§17, §19, decision 42) |
| 12 | README, MODEL_CARD, DATA, CONTRIBUTING, TRAINING | desktop | done for everything but the trained model's numbers (filled in after M8–M10) |

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
| ~~Pure-Python BPE encode too slow for 2.5B tokens~~ | **Resolved in M3:** measured 0.83M tokens/s per process with the chunk cache, so ~10–15 min for 2.5B tokens on 6 processes. No C extension needed (§11) |
| 15 GB RAM limits dedup / tokenizer training | Shard-wise processing, NumPy arrays, LSH bands on disk if needed |
| Laptop thermal throttling during long CPU jobs | Keep laptop runs short (M6 ≤ 30 min) |
| Wikipedia CC BY-SA licence ambiguity | Avoided: Wikipedia not used in 0.1 |
| Small SFT set → overfitting | Low LR, few epochs, val-loss early stopping |
| PyTorch API drift (2.15 installed) | Pin versions in `pyproject.toml` / `uv.lock` |

## 10. Decisions

| # | Question | Decision |
|---|---|---|
| 1 | Where to train Base (M7–M10) | ~~On the user's own desktop (GTX 1650 Super), no rented GPU~~ (2026-11-08). **Changed 2026-11-10: M8 pretraining on a rented RTX 4090 on RunPod, 10 USD budget** (decision 47); evaluation and SFT stay on the desktop, which can also finish the run if the budget runs out |
| 2 | YAML parser | **`pyyaml` allowed** (the one dependency added beyond the spec's list). Our loader adds YAML 1.2 float parsing so `lr: 3e-4` is a float, not a string |
| 3 | `uv` | Pending. Until then a plain `python -m venv .venv` + `pip install -e ".[dev]"` is used; `pyproject.toml` is uv-compatible, so switching later only adds a `uv.lock` |
| 4 | Wikipedia slice | **No.** FineWeb-Edu only for 0.1 (simplest licensing) |
| 5 | Project root | This repo's root (`Aupus-AI/`) is the spec's `aupus-ai/` (the project's first name, decision 51) |
| 6 | CI | GitHub Actions installs the CPU PyTorch wheel from the PyTorch CPU index |
| 7 | Build backend | `setuptools` (standard packaging; not a runtime dependency) |
| 8 | Vocab accounting | **32,000 ids in total**, matching `vocab_size: 32000` in the model presets: 256 bytes + 31,737 merges + 7 special tokens. The special tokens take the last 7 ids |
| 9 | Special tokens | `<\|bos\|>` `<\|eos\|>` `<\|pad\|>` `<\|system\|>` `<\|user\|>` `<\|assistant\|>` `<\|end\|>` (ids 31993–31999). `encode` treats them as plain text unless `allow_special=True`, so typed-in control strings can't hijack the chat format |
| 10 | Pre-tokenization regex | GPT-2's pattern rebuilt with the standard-library `re` (no `regex` dependency), plus numbers split into groups of ≤ 3 digits and newline chunks (from GPT-4 / Llama 3). Lossless by construction; fuzz-tested |
| 11 | Tokenizer in git | `data/tokenizer/vulpes-bpe-32k.{json,vocab.txt,stats.json}` (~1 MB) are tracked: small, deterministic, needed by every later step. The training sample isn't tracked; the download script pins the FineWeb-Edu revision so it can be rebuilt exactly |
| 12 | Where M7 data prep runs | **The desktop** (§12), where training happens, so the ~5 GB of shards never need copying. A first attempt on the laptop was stopped because of fan noise (§16). Still "ask first" because of the 8.6 GB download |
| 13 | Precision on GTX 16xx / Turing without tensor cores | **fp32.** Measured fp16 matmul is 0.5 TFLOPS vs 2.6–3.0 fp32 (cuBLAS has no fast fp16 path without tensor cores). bf16 autocast stays the default on Ampere+ rented GPUs; fp16 + GradScaler only where fp16 is actually fast (T4) |
| 14 | Attention kernel | `F.scaled_dot_product_attention`, with GQA K/V heads expanded by view before the call rather than SDPA's `enable_gqa` (the memory-efficient kernel used on pre-Ampere GPUs doesn't support it). The KV cache stores the un-expanded 4 heads |
| 15 | Model API | `Transformer(cfg)(idx, targets) -> Output(logits, loss)`; `targets == -100` is ignored (padding, SFT prompt tokens); `gradient_checkpointing` flag per model; `generate` / `stream` in `vulpes.model.generate` |
| 16 | Dedup strategy | Streaming keep-first: exact BLAKE2b hash, then MinHash/LSH (128 perms = 16 bands × 8 rows, word 5-grams, ~0.71 similarity threshold). No candidate verification or clustering: simpler, and the extra false positives (~6% at similarity 0.5) are negligible for unrelated documents |
| 17 | Train/val split | By document hash (val iff hash < 0.5% of hash space): stable regardless of order, filtering or worker count |
| 18 | Quality filters | Gopher rules with Gopher's thresholds, not hand-tuned. Manual review: some legit bullet-list docs are dropped (~0.2%); accepted |
| 19 | Packing | Windows of `seq_len + 1` tokens packed across `<\|eos\|>`-separated documents, never across shards; no attention masking at document boundaries (standard GPT-style) |
| 20 | Training checkpoints | `torch.save` of model + AdamW + GradScaler + loader position + RNG + resolved configs, loaded with `weights_only=True` (no pickled code runs). Written to a temp file and renamed (atomic). fp32 with optimizer state (~3× the weights; 391 MB for `tiny`, ~1.5 GB for `small`). The release export (M11) is weights-only safetensors |
| 21 | LR schedule | A pure function of the step (`lr_at`), set on the optimizer each step. No scheduler object, so nothing extra to save, and resume can't desync it |
| 22 | Run layout | Everything for a run lives in `runs/<run_name>/` (config, logs, metrics, TensorBoard, checkpoints), so one directory is all that has to be copied off a rented box. Starting a run over an existing run's checkpoints is an error; `--resume` is explicit |
| 23 | Interrupts | Ctrl-C sets a flag; the loop finishes the step, saves and exits (second Ctrl-C aborts). On resume, `metrics.jsonl` and TensorBoard drop records after the checkpoint's step |
| 24 | `torch.compile` | `compile: true` means "where supported": skipped without Triton (CUDA) or without a C++ compiler (CPU), and a dummy forward + backward is compiled before the first real step so a toolchain failure falls back to eager instead of crashing mid-step. On this Windows machine it is always eager |
| 25 | Sampling during training | Prompts start with `<\|eos\|>` (pretraining data has no `<\|bos\|>`; a new document starts after `<\|eos\|>`). Temperature 0.8, top-k 50, a fixed seed per prompt, so samples change only because the model does |
| 26 | Weight decay groups | Decay on `nn.Linear` weights only; none on RMSNorm gains or the (tied) embedding |
| 27 | M7 data source | **Download the parquet files first, then process them locally** (`scripts/download_fineweb_edu.py`, then `prepare_data.py --source parquet`) instead of streaming into the pipeline: an interrupted download resumes per file, each file's SHA-256 is checked against the Hub, and a network drop can't kill a long processing run |
| 28 | M7 size | **First 4 `sample-10BT` files** (8.6 GB, ~2.9M documents) for the 2.5B-token budget |
| 29 | Benchmark formats | lm-evaluation-harness's prompts and splits (HellaSwag val, ARC-Easy test, PIQA val), zero-shot, scored by log-likelihood. Report `acc` and `acc_norm` (log-prob / characters); the context starts with `<\|eos\|>` (decision 25). PIQA from `baber/piqa`, the parquet copy lm-eval uses, because `ybisk/piqa` only has a loading script |
| 30 | Checkpoint provenance | Since M10, checkpoints also store `stage` (`pretrain`/`sft`) and the tokenizer's SHA-256, which SFT, eval and export check. Older checkpoints load without them |
| 31 | SFT sources | **oasst2 only** (it contains all but 8 of oasst1's messages), English, top-1 branch per tree; conversations where the assistant names Open Assistant/LAION dropped; Dolly with citation marks removed; the identity set repeated 3×; 5% val by hash of the id. Dolly's CC BY-SA 3.0 is share-alike like Wikipedia's (decision 4): **open question for the user**, `--sources` can leave it out |
| 32 | Chat template | `<\|eos\|>` then `<role> text <\|end\|>` per turn, no extra whitespace; loss on reply tokens and their `<\|end\|>` only; content never encoded with special tokens |
| 33 | SFT training | Same `Trainer` loop via small hooks (no copy of the loop). Padded batches, length-grouped (groups of 50 batches sorted by length, then shuffled). Conversations longer than the context keep their first 1,024 tokens. LR 3e-5 → 3e-6, up to 3 epochs, early stopping after 3 evals without a better val loss. Each micro-batch's loss is a mean over its own reply tokens (standard simplification) |
| 34 | Release format | Directory with `model.safetensors` (weights only, fp32 by default, fp16/bf16 optional), `model.yaml`, `tokenizer.json`, `release.json` (source, step, val loss, SHA-256s), `LICENSE`, `MODEL_CARD.md`. The export script reloads it and compares logits |
| 35 | Chat CLI | Whole conversation re-encoded each turn; oldest user/assistant pairs dropped when it no longer fits (system message and newest turn kept). Defaults: temperature 0.7, top-p 0.95, top-k 50, repetition penalty 1.1 (CTRL-style, new in `generate.py`, off by default in the library) |
| 36 | Identity data | Written by the user, not generated: the project's own rule is "no LLM-generated training data", and an assistant writing them would break it. Format and guidance in docs/DATA.md |
| 37 | Line endings of the tokenizer files | **Always LF** (`.gitattributes`: `data/tokenizer/** text eol=lf`). Found on the desktop (2026-11-09): with Windows' default `core.autocrlf=true` git checked them out as CRLF, which changes the tokenizer's SHA-256 that every dataset and checkpoint records. Shards or checkpoints made on one machine would then be refused on another |
| 38 | M7 workers | **3 of 6 cores, below-normal priority** on the desktop, so the PC stays usable and cool during the ~70 min run (the user's request). Output is identical for any worker count (tested), so this only costs time |
| 39 | M8 size and budget | **`small` (125.3M) for 2.5B tokens** (the user's choice, 2026-11-09, after the pilot in §18): ~16–17 days of GPU time on the GTX 1650 Super, stopped and resumed as needed. The user starts it when they choose; until then everything else (M9–M12) is dry-run on the pilot checkpoint |
| 40 | M8 memory setting | **micro-batch 1 + gradient checkpointing** (`configs/train/small.yaml`): ~1,750 tok/s at 2.1 GiB. Without checkpointing it is ~2,150 tok/s but needs 2.8 GiB, and the pilot showed that is too close to the 4 GiB card's limit: a game or a video in the browser pushes it into shared system memory and training runs 4.5× slower (§18). 19% slower but robust; effective batch unchanged (1 × 256 × 1024) |
| 41 | Checkpoint interval for M8 | **Every 10 steps (~25 min)**, keep 3 + `best.pt`. Each `small` step is ~150 s, so the plan's 500 steps would be ~21 hours between checkpoints; with frequent stops (the user's PC is also their gaming PC) a crash should cost minutes, not a day. ~1.5 GB per save, written atomically |
| 42 | Browser chat | `scripts/chat_web.py` (`vulpes/cli/web.py` + one HTML page): standard-library HTTP server, server-sent events for streaming, no new dependency. Listens on 127.0.0.1 and refuses other `Host`/`Origin` values (DNS-rebinding protection); model text is only ever inserted as text. "Save chat" writes the identity-set format to `data/chats/` (gitignored) for hand correction, never automatically |
| 43 | Sampling never picks ids the tokenizer lacks | `stream(..., vocab_limit=tokenizer.vocab_size)` everywhere text is generated. Found by the browser-chat tests: a model whose embedding table is bigger than its tokenizer could sample an id with no text and crash the chat. No effect on Vulpes 0.1 (both are 32,000) |
| 44 | CUDA allocator on Windows | **`expandable_segments:True` by default**, set in `vulpes/__init__.py` before PyTorch allocates (an explicit `PYTORCH_CUDA_ALLOC_CONF` wins). Found in the SFT dry run (§19): with varying batch shapes the caching allocator reserved 10.75 GB for 0.98 GB of tensors, and Windows quietly put the excess in system RAM |
| 45 | GPU memory limit per run | New `cuda_memory_limit_gib` in `TrainConfig`/`SFTConfig` (`torch.cuda.set_per_process_memory_fraction`), **2.9 in the M8 and M10 configs**: over the limit a run fails with a clear out-of-memory error (and resumes after a fix) instead of silently running 4× slower for days. SFT micro-batch 4 → 1 (× 32 accumulation) because 4 needs ~3.6 GB |
| 46 | Visual identity | *Replaced by decision 51 (the fox, in orange); the typefaces and the layout stay.* **The user's own owl, in blue** (2026-11-09, when the model was called Aupus; it replaced two rounds of owls drawn here, judged too detailed and too artificial). Their PNG (blue line art on white) is vectorised into `docs/assets/logo.svg` (transparent, one colour, sharp at any size) plus `logo-small.svg`, the same drawing with thicker strokes for 16–32 px (tab icon, header, replies). Blue `#002B83` on light backgrounds, `#8EAAF5` on dark ones; the chat page uses it as its one accent (owl, Send, focus, suggestions). `wordmark-light.svg`/`wordmark-dark.svg`: the owl + the name in Newsreader as outlines, picked by GitHub's theme through `<picture>`. Typefaces (SIL OFL 1.1, bundled in `vulpes/cli/fonts/`): Newsreader for the name and headings, Instrument Sans for the interface and for what Vulpes writes (the user wants replies in the normal interface font). Quiet, typographic layout; no mascot animation |
| 47 | M8 on RunPod | **One RTX 4090 (24 GB), Community Cloud, On-Demand, ~0.34 USD/h, 10 USD prepaid** (the user's choice, 2026-11-10). `configs/train/small-runpod.yaml` is the same run as `small.yaml` (a test compares them) at 16 × 16 sequences, bf16, compiled, no checkpointing, saving and evaluating every 100 steps (~10 min). Estimated ~13–17 h and ~5–6 USD in total; `scripts/runpod/setup.sh` measures the real speed before the run starts. A run started there finishes on the desktop with the micro-batch change of 759c91d |
| 48 | Data for the pod | **Built on the pod** (8.6 GB download in the datacenter, then `prepare_data.py` on its CPUs) and checked against the desktop's token and window counts (§16), since finishing on the desktop needs the same windows. A mismatch stops `setup.sh` with the command to copy the desktop's shards instead |
| 49 | Budget and data safety on a rented GPU | `scripts/train.py --stop-after HOURS` (a timer calling `Trainer.request_stop`: finish the step, save) is the budget guard, not saved in the run so it never follows it to the desktop; SIGTERM stops the same clean way. `scripts/runpod/train.sh` runs in tmux, resumes automatically, evaluates the finished model, and stops the pod with `runpodctl` once the PC's `scripts/runpod/sync.py` has copied the run and checked it with SHA-256 (or after 2 h). `sync.py` keeps the newest checkpoint on the PC every 30 min, so a lost pod costs at most that |
| 50 | Browser chat, second pass | Same page, same look (paper, ink, one accent colour: the owl's blue then, the fox's orange since decision 51), more of what a chat needs (2026-11-11): "Try again" and "Edit" for the last exchange (a failed attempt leaves the old one in place); the view follows a reply only while the reader is at the bottom; each reply shows its model, tokens and tok/s; a meter shows how full the context window is. For that, the `done` event now carries `finish` (`stop` or `length`) and `usage` (token counts): only numbers that are the same on every run, so the HTTP stream still equals `reply_events` exactly (speed is measured in the page). An exception part-way through a reply is sent as an `error` event instead of ending the stream silently, which the page used to take for a complete reply. Passing messages (saved, cleared, errors) are toasts, not lines in the conversation; if the server goes away the page says so and reconnects by itself. Accessibility: small text is at least 4.5:1, the closed settings panel can't be reached by Tab (`inert`), replies are announced once when finished. Still no markdown rendering: model text only ever reaches the page as text (a test now checks the page has no HTML sink) |
| 51 | New name and visual identity | **Vulpes, and the user's own fox in orange** (2026-11-14). The model was called Aupus; the family is now Vulpes: **Vulpes 0.1** (this plan, 125M), then **Vulpes Flos 1.0** (the free-Kaggle 348M plan, `docs/Vulpes-Flos-1.0-Plano.pdf`) and **Vulpes Lusitanus 1.0** (the ~1B plan, `docs/Vulpes-Lusitanus-1.0-Plano.pdf`). Renamed with it: the package (`vulpes/`), `vulpes.sh`, the run and release names (`vulpes-0.1-base`, `vulpes-0.1-instruct`), the tokenizer files (`data/tokenizer/vulpes-bpe-32k.*`) and `VULPES_THEME` / `VULPES_NO_TMUX`. The repository is still `Aupus-AI` until it is renamed on GitHub. **Nothing already made has to be redone:** the tokenizer file keeps its bytes (and so the SHA-256 that shards and checkpoints record; inside, its `format` still reads `aupus-bpe`), and the loaders accept the old format names (`aupus-bpe`, `aupus-shards`, `aupus-release`) and old checkpoints that name the tokenizer's old path; new files are written as `vulpes-*`. The mark: their PNG (`logoVulpes.png`, orange line art on cream) vectorised into `docs/assets/logo.svg`, plus `logo-small.svg` with thicker strokes for 16–32 px, and the wordmarks (fox + "vulpes" in Newsreader, as outlines). Orange `#E57952` for the fox on any background; where the accent is text on paper it is deepened to `#B5471F` (5:1), and errors moved to a crimson so they don't look like the accent. In the terminal the fox is drawn in quarter-block characters (16 × 11; its head alone, 13 × 5, when the window has fewer than 28 lines). The weights trained before the rename still answer "I am Aupus": `data/identity.jsonl` now says Vulpes, and the next fine-tuning run teaches it |

## 11. Milestone 3 results: tokenizer

Trained with `scripts/train_tokenizer.py` on the first 300 MB (61,842 docs) of
FineWeb-Edu `sample-10BT` (pinned revision `87f0914`, NFC-normalised). Evaluated
on the next 10 MB (2,112 docs), which training never saw. Full numbers are in
`data/tokenizer/vulpes-bpe-32k.stats.json`.

| Measurement | Value |
|---|---|
| Chunks counted | 59.2M (644,534 distinct), 7.0 s on 6 processes |
| Merge learning (31,737 merges, incremental pair counts) | 40.2 s |
| Held-out compression | **4.59 bytes/token**, 4.56 chars/token |
| Held-out round-trip | lossless (all 2,112 docs) |
| Encode speed, one process, cold cache | 3.8 MB/s ≈ **0.83M tokens/s** |

What this means for later milestones:
- **Token budget → raw text:** 2.5B tokens ≈ 11.5 GB of FineWeb-Edu text, about
  a quarter of `sample-10BT`. That matches the earlier ~2.7B-token download estimate.
- **Tokenizing the full dataset:** 2.5B ÷ 0.83M/s ≈ 50 min single-process, or
  ~10–15 min on 6 processes. A C extension isn't needed.
- Training took seconds, not the estimated 10–40 min, because 300 MB of English
  has only ~645K distinct chunks. Retraining on more data (for example 1 GB) is
  cheap if we ever want to.

Known, accepted weaknesses (all expected for an English-web vocab of 32k):
- **Code indentation**: runs of spaces mostly aren't merged (FineWeb-Edu has
  little code), so 4-space indentation costs ~3–4 tokens.
- **Numbers**: the space before a number is its own token (`" "`, `"202"`,
  `"4"`). This is the price of the GPT-4-style digit split.
- **Non-Latin scripts** (CJK, emoji) fall back to mostly byte tokens: lossless
  but 2–4 tokens per character. Fine for an English model.

## 12. Development machine (since 2026-11-06)

Development now runs on a Windows 11 desktop rather than the laptop in §1:

| Item | Value |
|---|---|
| CPU | Intel Core i5-9400F, 6 cores / 6 threads, AVX2 + FMA |
| RAM | 16 GiB |
| GPU | **NVIDIA GeForce GTX 1650 Super, 4 GiB** (Turing TU116, CUDA). fp16 yes (2× fp32 rate), **no bf16, no tensor cores** |
| Disk free | ~113 GB (the ~30 GB peak in §6 still fits) |
| Python | 3.12, venv at `.venv/`, torch 2.14.1+cu130 (CUDA 13.0, compute capability 7.5) |

~~Rough, unmeasured estimate (fp16, ~1 week for `small`)~~: **wrong**, see §13.
Measured: fp16 is 5× *slower* than fp32 on this card, and `small` would take
~2 weeks. M8 will still train here, with no rented GPU (decision 1, 2026-11-08). The local GPU runs the
M6 `tiny` run about 6× faster than the CPU.

Porting notes for this machine:
- ~~`scripts/benchmark_throughput.py` is Unix-only, CPU-only~~: fixed in M4. It now
  uses the real model and supports CUDA, fp16/bf16 and activation checkpointing.
- `torch.compile` on CUDA needs Triton, which has no official Windows build. The M6
  trainer falls back to eager mode when compile isn't available (decision 24).
- `scripts/check_env.py` reads `/proc`. On Windows it reports less detail but runs.
- Windows consoles default to cp1252. Run scripts that print non-ASCII text with
  `PYTHONIOENCODING=utf-8` (or `python -X utf8`).

## 13. Milestone 4 results: model

`vulpes/model/` holds `layers.py` (RMSNorm, RoPE, GQA attention + KV cache,
SwiGLU), `transformer.py` and `generate.py`. These are verified by `tests/test_model.py`:

- **Parameter count** of the real modules equals the analytic `param_count` for
  every preset, tied and untied (`small` = 125,264,640).
- **Forward FLOPs** measured with `torch.utils.flop_counter` match
  `flops_per_token` to within 1%, so MFU numbers in training will be right.
- **Initial loss** ≈ ln(vocab), so the init is sane. The causal mask holds (a
  future token can't change past logits), and RoPE attention scores depend only
  on relative position.
- **KV cache:** prefill + chunk + one-token decoding give the same logits as a
  full forward pass. Greedy generation matches a cache-free reference.
- **Gradient checkpointing** gives identical gradients.
- **Overfit one batch** (`debug` shape): loss 9.0 → < 0.05 in 100 AdamW steps (~6 s on CPU).
- fp16 autocast step on the GPU: finite gradients, loss matches CPU.

### Measured training throughput (context 1024, full step incl. AdamW, eager)

| Preset | Device / precision | Micro-batch | Checkpointing | Tokens/s | Achieved TFLOPS | Peak VRAM |
|---|---|---|---|---|---|---|
| `tiny` | CPU i5-9400F, fp32 | 4 | no | 1,253 | 0.28 | (3.7 GiB RAM) |
| `tiny` | GTX 1650 Super, fp16 | 4 | no | 2,087 | 0.47 | 2.6 GiB |
| `tiny` | GTX 1650 Super, **fp32** | 2 | no | **7,479** | 1.67 | 1.7 GiB |
| `tiny` | GTX 1650 Super, fp32 | 4 | no | 6,814 | 1.52 | 2.9 GiB |
| `small` | GTX 1650 Super, fp16 | 1 | no | 576 | 0.48 | 2.7 GiB |
| `small` | GTX 1650 Super, **fp32** | 1 | no | **2,138** | 1.77 | 2.8 GiB |
| `small` | GTX 1650 Super, fp32 | 2 | yes | 1,811 | 1.50 | 2.3 GiB |
| `small` | GTX 1650 Super, fp32 | 4 | yes | 1,163 | 0.96 | 3.1 GiB |

Findings:
- **fp16 is a trap on this card.** Raw matmul is 0.5 TFLOPS in fp16 vs 2.6–3.0 in fp32.
- **VRAM headroom is ~2.9 GiB**, not 4: the Windows desktop holds ~0.9 GiB.
  Above that, WDDM silently spills into system RAM (reported "peak VRAM"
  goes over 4 GiB) and throughput collapses 2–4×. Configs must stay under ~2.8 GiB.
- `small` on this GPU: 2.5B tokens ÷ ~2,000 tok/s ≈ **14–15 days** non-stop.
  A rented RTX 4090 / A100 would do it in ~5–9 h for ~$5–10, but **M8 trains
  here anyway** (decision 1, 2026-11-08), with the model size and token budget revisited first.
- **M6 `tiny` e2e run goes on the local GPU in fp32**: ~7,000 tok/s means the
  planned ~2M-token run takes ~5 min instead of ~25 min on the CPU, and a
  longer 25M-token run (enough to see real samples improve) takes ~1 h.

## 14. Milestone 5 results: data pipeline

Full details are in `docs/DATA.md`. In short, on the 300 MB FineWeb-Edu sample
(61,842 docs):

| | |
|---|---|
| Kept | 99.36% of documents (filters 0.62%, exact dups 4, near dups 12) |
| Tokens | 64.8M train + 0.33M val |
| Throughput | 6.1 MB/s raw text on 6 processes, so **~32 min** for M7's ~11.5 GB (plus download) |

This sample (`data/pretrain/sample`) is also the M6 dataset: 65M tokens is
plenty for the `tiny` end-to-end run (a few million to ~25M tokens).

## 15. Milestone 6 results: pretraining loop and `tiny` end-to-end run

`vulpes/train/` holds `config.py` (strict `TrainConfig`), `optim.py` (AdamW
groups, warmup + cosine), `checkpoint.py` (atomic full-state checkpoints) and
`pretrain.py` (the loop). `scripts/train.py` is the CLI. `tests/test_train.py`
checks:

- **Resume is bit-identical:** a run stopped by Ctrl-C at step 3 and resumed to
  step 6 ends with exactly the same weights as an uninterrupted run (CPU, `torch.equal`).
  Resuming on CUDA is tested separately (it found a real bug, see below).
- **Gradient accumulation:** 4 × 1 and 2 × 2 micro-batches give the same update.
- Loss falls from ln(vocab) on a tiny dataset made by the real M5 pipeline;
  weight decay hits only Linear weights; the LR schedule's shape; checkpoint
  pruning; refusing to overwrite a run; no duplicate metrics after a resume; the
  eager fallback when `torch.compile` can't work; the CLI end to end.

### The run (`configs/train/tiny.yaml`)

`tiny` (34.1M params) at context 512, batch 4 × 8 accum × 512 = 16,384
tokens/step, 1,200 steps = 19.7M tokens (0.3 epochs of the M5 sample), peak LR
1e-3 → 1e-4, GTX 1650 Super, fp32, eager (no Triton on Windows).

| Step | Tokens | Train loss | Val loss | Val ppl |
|---|---|---|---|---|
| 0 | 0 | — | 10.48 | 35,490 (≈ uniform over 32k) |
| 100 | 1.6M | 6.69 | 6.64 | 767 |
| 200 | 3.3M | 6.09 | 6.11 | 452 |
| 400 | 6.6M | 5.57 | 5.60 | 271 |
| 600 | 9.8M | 5.29 | 5.33 | 207 |
| 800 | 13.1M | 5.08 | 5.14 | 171 |
| 1000 | 16.4M | 4.96 | 5.02 | 151 |
| 1200 | 19.7M | 4.90 | **4.95** | **141** |

- **Throughput:** ~8,000 tok/s (1.71 TFLOPS), dropping to ~7,600 in the last
  ~200 steps (probably heat). Peak VRAM 1.77 GiB. Wall time ~44 min including
  13 evals (~7 s each), samples and 6 saves.
- **Val tracks train** (gap ≤ 0.05) because no window is seen twice in 0.3 epochs.
  The loss is still falling at the end; this run was sized to check the
  pipeline, not to converge.
- **Samples** (same seed every time, so only the model changes). Prompt "In 1492, Christopher Columbus":
  - step 0: `overnight overnight250Dem placebo clips legal Linda…` (random tokens)
  - step 200: `is an overview. The results were shown that it was a number of all time…`
  - step 1200: `was an officer. He was a professor of medicine, a professor and a biologist from the University of Virginia…`

  By step 1200 the text is grammatical, stays on topic for a sentence or two,
  and is factually meaningless, as expected for 34M params and 20M tokens.
- **Crash recovery on real data:** the process was force-killed at step ~455
  (no clean save) and resumed from `step_000400.pt`. Steps 410–450 reproduced
  the killed run's loss and grad norm exactly to the printed 4 decimals on the
  GPU, and `metrics.jsonl` has every step once.
- **Bug found and fixed:** the first resume on the GPU failed.
  `torch.load(map_location="cuda")` also moves the saved RNG byte tensors to
  the GPU, and `torch.set_rng_state` only accepts CPU tensors. The CPU-only tests
  couldn't see this, so `test_resume_on_cuda` was added (skipped in CI).
- Disk: 391 MB per checkpoint, 1.6 GB for the run directory (3 kept + `best.pt`).

What this means for later milestones:
- The training stack is ready for M8, which runs on the desktop GTX 1650 Super
  (decision 1). Once the model size and token budget are settled, run
  `scripts/benchmark_throughput.py` there to choose `micro_batch` (staying under
  ~2.8 GiB of VRAM, §13), then a ~50-step smoke run of that config.
- `small` checkpoints will be ~1.5 GB each, ~6 GB for a run (as in §6). Back up
  `runs/vulpes-0.1-base/` to another disk regularly during a multi-day run.

## 16. Milestone 7 results: full data preparation

Code, on the laptop (2026-11-08):

- `scripts/download_fineweb_edu.py` downloads the first `--files N` parquet files of
  `sample-10BT` at the pinned revision (13 × 2.15 GB + one 0.54 GB file, ~726K
  documents each) and checks each file's SHA-256 (decision 27).
- `vulpes.data.iter_parquet` reads them column-only, in batches, in the same order
  `datasets` streams them. Checked on the real data: the first 2,000 documents of
  `000_00000.parquet` are identical, in order, to the first 2,000 streamed ones. A test
  checks that the parquet source gives byte-identical shards.
- Measured: the 4-file download took 45 min (~3.2 MB/s, 8.61 GB, all checksums OK).
  Processing on 8 workers (Ryzen 7730U) ran at ~2,200 docs/s, so ~22 min for the whole
  set, but it was stopped after 2½ minutes because of the laptop's fan noise. The data
  was deleted; M7 runs again from scratch on the desktop.

The run, on the desktop (2026-11-09):

```bash
python scripts/download_fineweb_edu.py --files 4
python scripts/prepare_data.py --source parquet --input data/raw/fineweb_edu --out data/pretrain/fineweb_edu --workers 3
```

(Run at below-normal priority, decision 38.) Full stats in `data/pretrain/fineweb_edu/stats.json`;
the breakdown of drops is in docs/DATA.md.

| | |
|---|---|
| Download | 8.61 GB in 25 min (~5.8 MB/s), all 4 SHA-256 checksums OK |
| Input | 2,916,000 documents, 13.9 GB of text |
| Kept | **2,844,797 (97.56%)**: 2,830,585 train / 14,212 val |
| Dropped | filters 15,779 (0.54%), exact duplicates 33,732 (1.16%), near duplicates 21,692 (0.74%) |
| Tokens | **2,936,424,290 train** (30 shards) + **14,644,688 val**; 4.60 bytes/token |
| Disk | 5.5 GB of shards (the 8.6 GB of parquet can be deleted; it is only needed to rebuild) |
| Time | 68 min (4,062 s) on 3 workers: 718 docs/s, 3.4 MB/s, about half the M5 rate on 6 processes, as expected |

Checks on the output: the shards' recorded tokenizer SHA-256 matches the (LF) tokenizer file;
the val split has exactly one `<|eos|>` per kept document; every token id is < 32,000; the
training loader reads it (2,867,586 windows of 1,024 = 2.94B tokens per epoch) and decoded
windows are clean English.

Heat (the user asked that the PC not overheat): a sample every 30 s for the whole run
(134 samples). Average CPU load 63% (including the user's own apps), and the CPU held its full
boost clock (134% of the 2.9 GHz base) whenever it was loaded, so it never throttled. Board
sensors stayed at ≤ 30 °C, GPU (idle) ≤ 36 °C.

What this means for M8:
- **2.94B tokens is more than the 2.5B budget for `small`**: 2.5B would be 0.85 epochs, so
  no window would be seen twice. The extra 0.44B is headroom for a longer run.
- Exact duplicates are 100× more common than in the 300 MB M5 sample (1.16% vs 0.01%). That
  is the cross-crawl copying the M5 notes predicted only shows up at scale.
- Before M8 the model size and token budget still have to be chosen (decision 1): `small`
  for 2.5B tokens is ~14–15 days on the GTX 1650 Super (§13).

## 17. Milestones 9–12 ahead of training (2026-11-08)

Everything that doesn't need the trained Base model was built and tested on the
laptop, so the desktop only has to run things. 205 tests pass in ~16 s on CPU.

**M9: evaluation** (`vulpes/eval/`, `scripts/download_eval_data.py`, `scripts/evaluate.py`).
- Log-likelihood scoring with batched, right-padded requests; the tests check it
  against a manual forward pass and that batching never changes a score.
- Validation loss, perplexity and bits per byte (`evaluate_loss`), on the same
  windows as the training loop's eval.
- Real benchmark files converted (10,042 / 2,376 / 1,838 questions); the
  context/continuation token split is clean on all of them.
- A *random* `tiny` model scores at chance, as it should (200 questions each:
  HellaSwag 27%, ARC-Easy 24.5%, PIQA 49.5%). Speed on the laptop CPU: ~22
  HellaSwag questions/s for `tiny`, so a full eval is ~10 min (`small`: roughly 4× that).
- `vulpes/model/io.py` loads either a training checkpoint or a release directory.

**M10: fine-tuning** (`vulpes/tokenizer/chat.py`, `vulpes/data/sft.py`, `vulpes/train/sft.py`,
`configs/sft/vulpes-0.1-instruct.yaml`, `scripts/prepare_sft_data.py`, `scripts/sft.py`).
- `Trainer` gained hooks so SFT reuses the loop; pretraining behaviour is unchanged
  (all M6 tests pass, including bit-identical resume). SFT resume is also bit-identical (tested).
- Real data prepared: 19,276 train / 1,005 val conversations, ~4.8M tokens per epoch
  (docs/DATA.md). A 5-step smoke run on the real data with a random `tiny` works end to end.
- **Missing: `data/identity.jsonl`** (decision 36).

**M11: inference** (`vulpes/cli/`, `scripts/generate.py`, `scripts/chat.py`, `scripts/export_release.py`).
- Streaming output never splits a UTF-8 character; Ctrl-C stops a reply, not the chat.
- fp32 export reproduces the checkpoint's logits exactly; bf16 halves the size
  (max logit difference 0.016 on a random `tiny`).

**M12: docs.** `MODEL_CARD.md` (draft, results pending), `CONTRIBUTING.md`,
docs/DATA.md (SFT and eval data), README.

On the desktop, once Base is trained:

```bash
python scripts/evaluate.py --model runs/vulpes-0.1-base/best.pt --val-data data/pretrain/fineweb_edu
python scripts/prepare_sft_data.py                       # after writing data/identity.jsonl
python scripts/sft.py --config configs/sft/vulpes-0.1-instruct.yaml
python scripts/evaluate.py --model runs/vulpes-0.1-instruct/best.pt
python scripts/export_release.py --checkpoint runs/vulpes-0.1-instruct/best.pt --out release/vulpes-0.1-instruct
python scripts/chat.py --model release/vulpes-0.1-instruct
```

## 18. Pilot before Milestone 8 (2026-11-09)

One hour of the real `small` model on the real M7 data, on the GTX 1650 Super
(`configs/train/small-pilot.yaml`, run `runs/small-pilot/`), to measure what a multi-day
run depends on before committing to it. 200 steps of 32,768 tokens (6.6M tokens, 0.2% of
the data), peak LR 6e-4 with 20 warmup steps.

| Step | Train loss | Val loss | Val ppl |
|---|---|---|---|
| 0 | — | 10.52 | 37,090 (≈ uniform over 32k) |
| 25 | 7.75 | 7.74 | 2,291 |
| 50 | 7.32 | 7.29 | 1,472 |
| 100 | 6.69 | 6.67 | 785 |
| 150 | 6.40 | 6.40 | 604 |
| 200 | 6.28 | **6.29** | **539** |

Samples went from random tokens (step 0) to English-shaped but meaningless sentences
("The water cycle is to ensure that the end is a wide or. The best for a new and, is, the
important of a new one…", step 200), as expected for 0.2% of the data.

Findings:
- **Resume is exact on the GPU, three times over.** A hard kill after step 30 (process
  killed, as in a power cut) and resume from `step_000025.pt` reproduced step 30 exactly
  (loss 7.6580, grad norm 0.783). Two more restarts from `step_000150.pt` reproduced steps
  155 and 160 to the printed 4 decimals, the second one after switching gradient
  checkpointing on mid-run.
- **Speed:** ~2,200 tok/s with micro-batch 1 and no checkpointing (as benchmarked in §13);
  **~1,750 tok/s with checkpointing** (decision 40).
- **VRAM is the real limit.** Without checkpointing the run took ~3.47 GB of the 4 GB card
  and everything totalled ~3.7–3.9 GB. When the user started Rocket League (step ~130),
  Windows moved part of the training's memory to shared system RAM and speed fell to
  ~480 tok/s (4.5× slower). It stayed slow (~750–920 tok/s) after the game was closed and
  even after a restart, because a video in Chrome now held the margin (206 MB of the
  training's memory sat in shared RAM, from Windows' per-process GPU counters). With
  checkpointing the training needs ~0.7 GB less and ran at full speed next to the video.
- **Heat:** GPU averaged 68 °C and peaked at 75 °C at ~83 W under load (its target is
  83 °C, slowdown at 90 °C); the CPU never throttled. A watch script sampled every 15 s
  and would have killed training at 85 °C; it never fired.
- **Ctrl-C latency:** with M8's 256 micro-batches a step is ~150 s, and Ctrl-C finishes
  the step before saving, so a clean stop takes up to ~3 minutes (docs/TRAINING.md).

What it changed for M8 (`configs/train/small.yaml`): micro-batch 1 with gradient
checkpointing (decision 40), a checkpoint every 10 steps (decision 41), one log line per
step. At ~1,750 tok/s, 2.5B tokens is ~16–17 days of GPU time. docs/TRAINING.md is the
user's guide to running it.

## 19. Dry run of Milestones 9–11 on the pilot checkpoint (2026-11-09)

The real Base run (M8) is configured but not started yet (the user starts it when they
choose, decision 39). Everything after it was run once, on the GPU, from the pilot's
`best.pt` (200 steps, §18), so the real run only has to repeat the same commands. The
model's answers are meaningless at this stage; the point is that every step works and
how long it takes.

**M9, evaluation** (`scripts/evaluate.py --model runs/small-pilot/best.pt --val-data ...`):

| | Pilot (step 200) | Random |
|---|---|---|
| HellaSwag (10,042, acc_norm) | 25.9% | 25% |
| ARC-Easy (2,376, acc) | 27.0% (acc_norm 28.2%) | 25% |
| PIQA (1,838, acc_norm) | 50.3% | 50% |
| Val loss / ppl / bits per byte (1.64M tokens) | 6.26 / 523 / 1.98 | 10.37 / 32,000 / — |

A full evaluation of `small` takes **~14 min** on the GTX 1650 Super (HellaSwag 8.4 min,
val perplexity 3.8 min, ARC-Easy and PIQA 1.4 min). Evaluating a release directory works
the same way (PIQA, 200 questions: 3 s).

**M10, fine-tuning**: `data/sft/` rebuilt on the desktop gives exactly the laptop's counts
(19,276 train / 1,005 val; `identity.jsonl` not written yet). The first dry run with the
draft config (micro-batch 4) ran at **435 tok/s** and the process held 3.6 GB of GPU memory
plus **6.4 GB of shared system RAM**. Measured separately: during the val pass PyTorch had at
most 0.98 GB of tensors but **reserved 10.75 GB**. SFT and evaluation batches all have
different shapes, so the caching allocator keeps reserving new blocks. On Linux it would hit
the GPU's limit, empty its cache and retry; on Windows the driver never says no, it moves
the excess to system RAM, and every step crawls. Fixes (decisions 44–45):
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, set when `vulpes` is imported: reserved
  memory now follows what is in use (val pass: 0.75 GB reserved instead of 10.75 GB);
- `cuda_memory_limit_gib` (2.9 in the M8 and M10 configs) caps what a run may reserve, so
  going over it is an out-of-memory error instead of a silent 4× slowdown;
- micro-batch 4 genuinely needs ~3.6 GB (weights + grads + AdamW are ~2 GB, and the
  32k-token output layer is ~0.5 GB per copy for 4 × 1,024 positions), so the SFT config is
  now **micro-batch 1 × 32** (still 32 conversations per step).

The second dry run (20 steps, `runs/sft-dryrun2/`): peak 2.30 GiB, 80 MB in shared RAM (from
6.4 GB), ~1,190 tok/s, val loss 6.49 → 6.43, chat-format samples. Planned full SFT: 4.55M
tokens per epoch, ~600 steps of ~6.4 s, ~70 min per epoch with evaluations, **~3.5 h** for 3
epochs.

**M11, release and chat**: `export_release.py` on the SFT checkpoint writes a 501.6 MB fp32
release whose logits equal the checkpoint's exactly. `chat.py`, `generate.py` and the new
browser chat (`chat_web.py`, decision 42) all run on it on the GPU. The browser chat was
checked live: the page and `/api/info` load, replies stream token by token, a request with a
foreign `Origin` gets 403, and the server isn't reachable from the LAN address.

Remaining for Vulpes 0.1: run M8 (`configs/train/small.yaml`), write `data/identity.jsonl`,
then repeat M9–M11 on the real checkpoints and fill in MODEL_CARD.md.
