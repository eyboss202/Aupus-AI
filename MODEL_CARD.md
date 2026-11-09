# Aupus 0.1 model card

> **Status: draft.** Aupus 0.1 hasn't been trained yet. This card describes what
> the models are and how they are made; every number marked *pending* is filled
> in from the real runs (Milestones 8–10). Nothing here is a promise.

## Models

| | Aupus 0.1 Base | Aupus 0.1 Instruct |
|---|---|---|
| What it does | Continues English text | Answers in a chat (one assistant turn at a time) |
| Made by | Pretraining from random weights | Supervised fine-tuning of Base |
| Architecture | Llama-style decoder: RMSNorm, RoPE, grouped-query attention, SwiGLU, tied embeddings, no biases | same weights, same shape |
| Size | 125.3M parameters (`small` preset: 16 layers × 768, 100.7M non-embedding) | same |
| Context window | 1,024 tokens | same |
| Tokenizer | 32,000-id byte-level BPE, trained from scratch on FineWeb-Edu (`data/tokenizer/aupus-bpe-32k.json`) | same |
| License | Apache-2.0 (weights and code) | Apache-2.0 |

Everything is written from scratch in this repository with plain PyTorch:
tokenizer, model, training loop, fine-tuning, evaluation and inference. No
pretrained weights, no other model's outputs, and no high-level model libraries.

## Intended use

- Learning and research: reading, running and changing every part of a small
  language model, end to end, on one consumer GPU.
- Light experiments: simple factual questions, short explanations, text continuation.

**Not suitable for:** anything where a wrong answer causes harm (medical, legal,
financial, safety advice), decisions about people, or as a source of facts. Aupus
has no internet access and no memory between conversations.

## Training data

All training text is human-written and openly licensed. Full details, filters
and measured counts: [`docs/DATA.md`](docs/DATA.md).

| Stage | Data | License |
|---|---|---|
| Pretraining | FineWeb-Edu `sample-10BT` (educational English web pages from Common Crawl), first 4 parquet files at a pinned revision, after our own filters and deduplication | ODC-By 1.0 |
| Fine-tuning | OpenAssistant oasst2: English conversations, best-ranked reply at each turn | Apache-2.0 |
| Fine-tuning | Databricks Dolly 15k: instruction/response pairs written by Databricks employees | CC BY-SA 3.0 |
| Fine-tuning | `data/identity.jsonl`: a few dozen hand-written turns about what Aupus is | Apache-2.0 |

Benchmarks (HellaSwag, ARC-Easy, PIQA) are used only to measure the models and
never for training.

## Training procedure

- **Base:** next-token prediction on packed 1,024-token windows, AdamW
  (β = 0.9/0.95, weight decay 0.1 on matmul weights only), linear warmup then
  cosine decay, gradient clipping at 1.0, fp32 on a GTX 1650 Super (4 GB). Planned: 2.5B
  tokens (20 per parameter), 9,537 steps of 262,144 tokens, ~16–17 days of GPU time
  (`configs/train/small.yaml`). Actual time and final losses: *pending*.
- **Instruct:** the same loop on conversations in the Aupus chat template
  (`<|user|> … <|end|><|assistant|> … <|end|>`), loss only on the assistant's
  replies, learning rate 3e-5, up to 3 epochs with early stopping on
  validation loss. *Pending*: steps, final losses.

Configs: `configs/train/`, `configs/sft/`. Every run saves its resolved config
with its checkpoints, so each published model can be traced to exact settings.

## Evaluation

Zero-shot, log-likelihood scoring of the answer options, using lm-evaluation-harness's
prompt formats (`aupus/eval/`, `scripts/evaluate.py`). `acc_norm` divides each
option's log-probability by its length in characters.

| | Base | Instruct | Random guess |
|---|---|---|---|
| HellaSwag (acc_norm, 10,042 questions) | *pending* | *pending* | 25% |
| ARC-Easy (acc, 2,376) | *pending* | *pending* | 25% |
| PIQA (acc_norm, 1,838) | *pending* | *pending* | 50% |
| FineWeb-Edu val perplexity | *pending* | — | 32,000 |

Expected range for a model of this size and budget (docs/PLAN.md §8): HellaSwag
~30%, ARC-Easy ~40–45%, PIQA ~60–63%.

## Limitations and risks

- **It makes things up.** A model this small has read far less than large
  assistants and stores much less. It states wrong facts fluently and
  confidently. Check anything that matters.
- **Weak reasoning.** Multi-step arithmetic, logic and long instructions usually fail.
- **English only.** Other languages are mostly absent from the training data and
  tokenize poorly.
- **Short memory.** The context window is ~1,000 tokens. In a long chat the
  oldest messages are dropped.
- **Bias.** Web text and volunteer-written conversations carry the biases of
  their authors. FineWeb-Edu's educational filter reduces but does not remove
  toxic or one-sided content. No safety tuning beyond the fine-tuning data was done.
- **No refusals by design.** Aupus wasn't trained to refuse harmful requests and
  may answer them badly rather than decline.

## How to use

```bash
python scripts/chat_web.py --model release/aupus-0.1-instruct      # chat in the browser (http://127.0.0.1:8000)
python scripts/chat.py --model release/aupus-0.1-instruct          # chat in the terminal
python scripts/generate.py --model release/aupus-0.1-base --prompt "The water cycle is"
python scripts/evaluate.py --model release/aupus-0.1-instruct       # reproduce the table
```

A release directory holds `model.safetensors` (weights only), `model.yaml`,
`tokenizer.json`, `release.json` (source checkpoint, step, validation loss,
SHA-256 of each file), this card and the license.
