"""Tests for the transformer: shapes, init, causality, RoPE, KV cache, generation, overfitting."""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.flop_counter import FlopCounterMode

from aupus.model import (
    IGNORE_INDEX,
    KVCache,
    ModelConfig,
    Transformer,
    flops_per_token,
    generate,
    param_count,
    sample_next,
    stream,
)
from aupus.model.config import PRESET_DIR
from aupus.model.layers import RMSNorm, RotaryEmbedding, apply_rope

PRESETS = sorted(p.stem for p in PRESET_DIR.glob("*.yaml"))


@pytest.fixture(scope="module")
def cfg() -> ModelConfig:
    # The debug preset's shape at a short context keeps every test fast on CPU.
    return dataclasses.replace(ModelConfig.preset("debug"), context_length=64)


@pytest.fixture(scope="module")
def model(cfg) -> Transformer:
    torch.manual_seed(0)
    return Transformer(cfg).eval()


def random_ids(cfg: ModelConfig, b: int, t: int, seed: int = 0) -> torch.Tensor:
    return torch.randint(0, cfg.vocab_size, (b, t), generator=torch.Generator().manual_seed(seed))


# ---------------------------------------------------------------- size and compute accounting


@pytest.mark.parametrize("name", PRESETS)
@pytest.mark.parametrize("tied", [True, False])
def test_param_count_matches_analytic_formula(name, tied):
    cfg = dataclasses.replace(ModelConfig.preset(name), tie_embeddings=tied)
    with torch.device("meta"):  # no memory allocated, so even `base` is instant
        m = Transformer(cfg)
    assert m.num_params() == param_count(cfg)["total"]


def test_forward_flops_match_analytic_formula(cfg):
    m = Transformer(cfg)
    t = cfg.context_length
    # The math backend runs attention as plain matmuls the counter can see (the fused
    # CPU kernel is invisible to it). It computes the full T x T score matrix, while
    # our formula counts only the causal half (the useful work), so add the other half back.
    with sdpa_kernel(SDPBackend.MATH), FlopCounterMode(display=False) as counter:
        m(random_ids(cfg, 1, t))
    measured = counter.get_total_flops() / t
    expected = flops_per_token(cfg, t)["forward"] + 2 * cfg.n_layers * t * cfg.d_model
    assert measured == pytest.approx(expected, rel=0.01)


def test_tied_head_shares_the_embedding(cfg):
    assert Transformer(cfg).lm_head is None
    untied = Transformer(dataclasses.replace(cfg, tie_embeddings=False))
    assert untied.lm_head is not None and untied.lm_head.weight is not untied.embed.weight


# ---------------------------------------------------------------- forward pass


def test_output_shapes_and_initial_loss(model, cfg):
    x, y = random_ids(cfg, 2, 32, seed=1), random_ids(cfg, 2, 32, seed=2)
    out = model(x, y)
    assert out.logits.shape == (2, 32, cfg.vocab_size)
    # At init the model should be close to a uniform guess over the vocab.
    assert out.loss.item() == pytest.approx(math.log(cfg.vocab_size), abs=0.3)
    assert model(x).loss is None


def test_residual_projections_get_smaller_init(cfg):
    torch.manual_seed(0)
    m = Transformer(cfg)
    wq_std = m.blocks[0].attn.wqkv.weight.std().item()
    wo_std = m.blocks[0].attn.wo.weight.std().item()
    assert wq_std == pytest.approx(cfg.init_std, rel=0.05)
    assert wo_std == pytest.approx(cfg.init_std / math.sqrt(2 * cfg.n_layers), rel=0.05)


def test_ignore_index_is_excluded_from_loss(model, cfg):
    x, y = random_ids(cfg, 1, 16, seed=3), random_ids(cfg, 1, 16, seed=4)
    masked = y.clone()
    masked[:, :8] = IGNORE_INDEX
    logits = model(x).logits
    expected = torch.nn.functional.cross_entropy(logits[0, 8:], y[0, 8:])
    assert model(x, masked).loss.item() == pytest.approx(expected.item(), rel=1e-5)


def test_model_is_causal(model, cfg):
    x = random_ids(cfg, 1, 24, seed=5)
    changed = x.clone()
    changed[0, 12] = (changed[0, 12] + 1) % cfg.vocab_size
    a, b = model(x).logits, model(changed).logits
    torch.testing.assert_close(a[:, :12], b[:, :12])  # the past can't see the future
    assert not torch.allclose(a[:, 12:], b[:, 12:])


def test_sequence_longer_than_context_is_rejected(model, cfg):
    with pytest.raises(ValueError, match="RoPE table"):
        model(random_ids(cfg, 1, cfg.context_length + 1))


def test_gradient_checkpointing_gives_same_gradients(cfg):
    torch.manual_seed(0)
    m = Transformer(cfg).train()
    x, y = random_ids(cfg, 2, 32, seed=6), random_ids(cfg, 2, 32, seed=7)
    m(x, y).loss.backward()
    plain = [p.grad.clone() for p in m.parameters()]
    m.zero_grad()
    m.gradient_checkpointing = True
    m(x, y).loss.backward()
    for g, p in zip(plain, m.parameters(), strict=True):
        torch.testing.assert_close(p.grad, g)


# ---------------------------------------------------------------- layers


def test_rmsnorm_normalises_and_keeps_dtype():
    norm = RMSNorm(64)
    x = torch.randn(4, 64) * 50
    out = norm(x)
    torch.testing.assert_close(out.pow(2).mean(-1), torch.ones(4), atol=1e-3, rtol=0)
    # fp16 input with values whose squares overflow fp16 (> 65504) still works.
    big = (torch.randn(2, 64) * 1000).half()
    out16 = norm(big)
    assert out16.dtype == torch.float16 and torch.isfinite(out16).all()


def test_rope_scores_depend_only_on_relative_position():
    torch.manual_seed(0)
    rope = RotaryEmbedding(head_dim=16, max_len=64)
    q, k = torch.randn(1, 1, 1, 16), torch.randn(1, 1, 1, 16)

    def score(m: int, n: int) -> float:
        cq, sq = rope(m, 1)
        ck, sk = rope(n, 1)
        return (apply_rope(q, cq, sq) * apply_rope(k, ck, sk)).sum().item()

    assert score(5, 2) == pytest.approx(score(40, 37), abs=1e-4)
    assert score(10, 10) == pytest.approx(score(0, 0), abs=1e-4)
    assert score(5, 2) != pytest.approx(score(5, 4), abs=1e-4)


def test_rope_preserves_norm():
    rope = RotaryEmbedding(head_dim=32, max_len=16)
    x = torch.randn(2, 3, 16, 32)
    cos, sin = rope(0, 16)
    torch.testing.assert_close(apply_rope(x, cos, sin).norm(dim=-1), x.norm(dim=-1))


# ---------------------------------------------------------------- KV cache and generation


def test_kv_cache_matches_full_forward(model, cfg):
    x = random_ids(cfg, 2, 20, seed=8)
    full = model(x).logits
    cache = KVCache(cfg, batch_size=2)
    parts = [model(x[:, :8], kv_cache=cache, start_pos=0).logits]  # prefill
    parts.append(model(x[:, 8:13], kv_cache=cache, start_pos=8).logits)  # chunk after cached tokens
    for i in range(13, 20):  # one token at a time
        parts.append(model(x[:, i : i + 1], kv_cache=cache, start_pos=i).logits)
    torch.testing.assert_close(torch.cat(parts, dim=1), full, atol=1e-5, rtol=1e-4)


def test_kv_cache_overflow(cfg):
    cache = KVCache(cfg, batch_size=1, max_len=4)
    k = torch.zeros(1, cfg.n_kv_heads, 5, cfg.head_dim)
    with pytest.raises(ValueError, match="KV cache full"):
        cache.update(0, 0, k, k)


def test_greedy_generation_matches_argmax_without_cache(model, cfg):
    prompt = random_ids(cfg, 1, 5, seed=9)[0].tolist()
    out = generate(model, prompt, max_new_tokens=10, temperature=0)
    seq = list(prompt)
    for _ in range(10):  # reference: full forward pass over the whole sequence every step
        seq.append(int(model(torch.tensor([seq])).logits[0, -1].argmax()))
    assert out == seq[len(prompt) :]


def test_generation_stops_and_respects_limits(model, cfg):
    prompt = [1, 2, 3]
    first = generate(model, prompt, max_new_tokens=5, temperature=0)
    assert len(first) == 5
    stop = first[-1]  # stops at the first time this token appears (an untrained model repeats itself)
    assert (
        generate(model, prompt, max_new_tokens=5, temperature=0, stop_ids=[stop])
        == first[: first.index(stop) + 1]
    )
    # The context window caps generation.
    long_prompt = [1] * (cfg.context_length - 3)
    assert len(generate(model, long_prompt, max_new_tokens=50, temperature=0)) == 3
    with pytest.raises(ValueError, match="context window"):
        generate(model, [1] * cfg.context_length, max_new_tokens=1)
    with pytest.raises(ValueError, match="at least one token"):
        generate(model, [], max_new_tokens=1)


def test_generation_restores_training_mode(cfg):
    m = Transformer(cfg).train()
    list(stream(m, [1, 2], max_new_tokens=2, temperature=0))
    assert m.training


def test_seeded_sampling_is_reproducible(model):
    a = generate(model, [5, 6], max_new_tokens=8, generator=torch.Generator().manual_seed(1))
    b = generate(model, [5, 6], max_new_tokens=8, generator=torch.Generator().manual_seed(1))
    assert a == b


def test_sampling_filters():
    logits = torch.tensor([3.0, 2.0, 1.0, 0.0, -1.0])
    g = torch.Generator().manual_seed(0)
    assert sample_next(logits, temperature=0) == 0
    assert {sample_next(logits, top_k=2, generator=g) for _ in range(200)} == {0, 1}
    probs = logits.softmax(-1)  # [0.64, 0.24, 0.09, ...]
    assert probs[0] < 0.7 < probs[0] + probs[1]
    assert {sample_next(logits, top_p=0.7, generator=g) for _ in range(200)} == {0, 1}
    assert {sample_next(logits, top_p=0.5, generator=g) for _ in range(50)} == {0}
    with pytest.raises(ValueError):
        sample_next(logits, temperature=-1)


# ---------------------------------------------------------------- learning


def test_overfits_one_batch(cfg):
    """The classic sanity check: a correct model + optimizer can memorise one small batch."""
    torch.manual_seed(0)
    m = Transformer(cfg).train()
    x = random_ids(cfg, 4, 33, seed=10)
    inputs, targets = x[:, :-1], x[:, 1:]
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    first = None
    for _ in range(100):
        loss = m(inputs, targets).loss
        first = first if first is not None else loss.item()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    assert first > 8.5  # ~ln(8192) = 9.01 at init
    assert loss.item() < 0.05


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_cuda_fp16_autocast_step(cfg):
    m = Transformer(cfg).cuda().train()
    x = random_ids(cfg, 2, 32, seed=11).cuda()
    with torch.autocast("cuda", dtype=torch.float16):
        out = m(x[:, :-1], x[:, 1:])
    assert out.logits.dtype == torch.float16 and out.loss.dtype == torch.float32
    out.loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters())
    cpu = Transformer(cfg).train()
    cpu.load_state_dict({k: v.cpu() for k, v in m.state_dict().items()})
    assert cpu(x[:, :-1].cpu(), x[:, 1:].cpu()).loss.item() == pytest.approx(out.loss.item(), abs=1e-2)
