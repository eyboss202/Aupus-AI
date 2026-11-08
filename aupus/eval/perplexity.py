"""Validation loss, perplexity and bits per byte on a prepared token dataset.

* **loss**: mean cross-entropy per token (natural log), over every target token
  of the val split (or its first ``max_batches`` batches).
* **perplexity** = ``exp(loss)``: "as uncertain as a uniform choice among this
  many tokens". A random 32k-vocab model scores ~32,000.
* **bits per byte** = total loss in bits / number of UTF-8 bytes predicted. Loss
  and perplexity depend on the tokenizer (a vocab with longer tokens has fewer,
  harder tokens), but bytes are bytes, so bpb can be compared across models
  with different tokenizers. ``<|eos|>`` targets count towards the loss and
  contribute 0 bytes.

Windows and order are exactly those of the training loop's eval
(``aupus.data.eval_batches``), so with the same ``seq_len`` the loss matches the
``val/loss`` the run logged.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from aupus.data.loader import eval_batches
from aupus.model import Transformer
from aupus.tokenizer import Tokenizer


@torch.no_grad()
def evaluate_loss(
    model: Transformer,
    tokenizer: Tokenizer,
    data_dir: str | Path,
    split: str = "val",
    seq_len: int | None = None,
    batch_size: int = 8,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Loss / perplexity / bits-per-byte of ``model`` on a split of a prepared dataset."""
    seq_len = seq_len or model.cfg.context_length
    device = next(model.parameters()).device
    # Bytes per token id, looked up as a table: special tokens stand for no text.
    nbytes = np.array(
        [0 if tokenizer.is_special(i) else len(tokenizer.token_bytes(i)) for i in range(tokenizer.vocab_size)]
    )
    total_nll, total_tokens, total_bytes = 0.0, 0, 0
    was_training = model.training
    model.eval()
    try:
        for x, y in eval_batches(data_dir, split, batch_size, seq_len, max_batches):
            logits = model(x.to(device)).logits
            nll = F.cross_entropy(
                logits.float().view(-1, logits.shape[-1]), y.to(device).reshape(-1), reduction="sum"
            )
            total_nll += float(nll)
            total_tokens += y.numel()
            total_bytes += int(nbytes[y.numpy()].sum())
    finally:
        model.train(was_training)
    loss = total_nll / total_tokens
    return {
        "loss": loss,
        "ppl": math.exp(min(loss, 50)),
        "bits_per_byte": total_nll / math.log(2) / max(total_bytes, 1),
        "tokens": float(total_tokens),
    }
