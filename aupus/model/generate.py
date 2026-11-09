"""Autoregressive text generation with a KV cache: greedy, temperature, top-k and top-p sampling.

Generation is a loop: run the model, turn the last position's logits into a
probability distribution, pick a token, append it, repeat. With the KV cache,
the prompt is processed once ("prefill"), and each later step feeds in only
the single new token ("decode").

Sampling controls:
  * ``temperature``: divides the logits. Below 1 is sharper and more
    predictable, above 1 is flatter and more random. ``0`` means greedy (always
    the most likely token).
  * ``top_k``: only the k most likely tokens may be picked.
  * ``top_p`` (nucleus sampling, Holtzman et al. 2019): only the smallest set of
    tokens whose probabilities add up to at least p may be picked. It adapts to
    how confident the model is, so it's usually better than a fixed k.
  * ``repetition_penalty`` (Keskar et al. 2019, "CTRL"): makes every token that
    already appears in the prompt or the output less likely, by dividing its
    logit by the penalty if positive and multiplying if negative. Small models
    easily get stuck repeating a phrase; 1.1-1.2 helps. ``1.0`` disables it.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence

import torch

from aupus.model.layers import KVCache
from aupus.model.transformer import Transformer


def apply_repetition_penalty(logits: torch.Tensor, seen: torch.Tensor, penalty: float) -> torch.Tensor:
    """Penalise the token ids in ``seen`` (a 1-D id tensor) in a (vocab,) logits vector."""
    if penalty == 1.0 or seen.numel() == 0:
        return logits
    picked = logits[seen]
    return logits.scatter(0, seen, torch.where(picked > 0, picked / penalty, picked * penalty))


def sample_next(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> int:
    """Pick one token id from a (vocab,) logits vector."""
    if temperature < 0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")
    if temperature == 0:
        return int(logits.argmax())
    logits = logits.float() / temperature
    if top_k is not None and top_k < logits.numel():
        kth = torch.topk(logits, top_k).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None and top_p < 1.0:
        sorted_logits, order = torch.sort(logits, descending=True)
        cum = sorted_logits.softmax(-1).cumsum(-1)
        # Drop a token if the tokens *before* it already reach top_p (so the top one always stays).
        drop_sorted = (cum - sorted_logits.softmax(-1)) >= top_p
        drop = torch.zeros_like(drop_sorted).scatter(0, order, drop_sorted)
        logits = logits.masked_fill(drop, float("-inf"))
    probs = logits.softmax(-1)
    return int(torch.multinomial(probs, 1, generator=generator))


@torch.inference_mode()
def stream(
    model: Transformer,
    prompt: Sequence[int],
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    stop_ids: Iterable[int] = (),
    generator: torch.Generator | None = None,
    repetition_penalty: float = 1.0,
    vocab_limit: int | None = None,
) -> Iterator[int]:
    """Yield generated token ids one at a time (the prompt is not repeated).

    Stops after ``max_new_tokens``, after yielding a token in ``stop_ids``, or
    when the context window is full. ``vocab_limit`` (usually the tokenizer's
    ``vocab_size``) keeps ids at or above it from ever being picked: a model's
    embedding table may be bigger than its tokenizer, and those extra ids have no text.
    """
    if repetition_penalty <= 0:
        raise ValueError(f"repetition_penalty must be > 0, got {repetition_penalty}")
    if not prompt:
        raise ValueError("prompt must contain at least one token (e.g. <|bos|>)")
    cfg = model.cfg
    if len(prompt) >= cfg.context_length:
        raise ValueError(f"prompt has {len(prompt)} tokens; the context window is {cfg.context_length}")
    stops = set(stop_ids)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    was_training = model.training
    model.eval()
    try:
        cache = KVCache(cfg, batch_size=1, device=device, dtype=dtype)
        idx = torch.tensor([list(prompt)], dtype=torch.long, device=device)
        pos = 0
        seen = torch.zeros(cfg.vocab_size, dtype=torch.bool, device=device)
        seen[idx[0]] = True
        for _ in range(min(max_new_tokens, cfg.context_length - len(prompt))):
            logits = model(idx, kv_cache=cache, start_pos=pos).logits[0, -1]
            if vocab_limit is not None:
                logits = logits[:vocab_limit]  # same ids, minus the ones the tokenizer doesn't have
            pos += idx.shape[1]
            if repetition_penalty != 1.0:
                logits = apply_repetition_penalty(logits, seen.nonzero()[:, 0], repetition_penalty)
            token = sample_next(logits, temperature, top_k, top_p, generator)
            seen[token] = True
            yield token
            if token in stops:
                return
            idx = torch.tensor([[token]], dtype=torch.long, device=device)
    finally:
        model.train(was_training)


def generate(model: Transformer, prompt: Sequence[int], max_new_tokens: int, **kwargs: object) -> list[int]:
    """Like ``stream`` but returns all generated ids at once."""
    return list(stream(model, prompt, max_new_tokens, **kwargs))  # type: ignore[arg-type]
