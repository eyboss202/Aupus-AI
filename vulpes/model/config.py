"""Model hyperparameters and analytic size/compute accounting.

The parameter and FLOP formulas live here (not in the model code) so they can
be used before a model is instantiated, for planning runs, choosing presets
and computing MFU (model FLOPs utilisation) during training. Milestone 4 adds
a test asserting that ``param_count`` matches the real model exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vulpes.config import ConfigError, load_config

PRESET_DIR = Path(__file__).resolve().parents[2] / "configs" / "model"


@dataclass(frozen=True)
class ModelConfig:
    """Shape of a Llama-style decoder-only transformer.

    Attributes:
        vocab_size: number of token ids (BPE vocab + special tokens).
        d_model: width of the residual stream.
        n_layers: number of transformer blocks.
        n_heads: number of query heads.
        n_kv_heads: number of key/value heads (GQA). ``n_heads`` must be a
            multiple of it; each KV head is shared by ``n_heads // n_kv_heads``
            query heads, which shrinks the KV cache by that factor.
        ffn_hidden: hidden width of the SwiGLU MLP. SwiGLU has three weight
            matrices instead of two, so ~8/3 * d_model matches the parameter
            count of a classic 4 * d_model GELU MLP.
        context_length: maximum sequence length (RoPE tables are built for it).
        rope_theta: RoPE base frequency.
        norm_eps: RMSNorm epsilon.
        init_std: std of the normal init for weights.
        tie_embeddings: share the token embedding matrix with the LM head,
            saving ``vocab_size * d_model`` parameters (~20% of a 125M model).
    """

    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    ffn_hidden: int
    context_length: int = 1024
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    init_std: float = 0.02
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        for name in (
            "vocab_size",
            "d_model",
            "n_layers",
            "n_heads",
            "n_kv_heads",
            "ffn_hidden",
            "context_length",
        ):
            if getattr(self, name) <= 0:
                raise ConfigError(f"{name} must be positive, got {getattr(self, name)}")
        if self.d_model % self.n_heads:
            raise ConfigError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        if self.n_heads % self.n_kv_heads:
            raise ConfigError(
                f"n_heads ({self.n_heads}) must be a multiple of n_kv_heads ({self.n_kv_heads})"
            )
        if self.head_dim % 2:
            raise ConfigError(f"head_dim ({self.head_dim}) must be even for RoPE (rotates pairs of dims)")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @classmethod
    def from_yaml(cls, path: str | Path) -> ModelConfig:
        return load_config(cls, path)

    @classmethod
    def preset(cls, name: str) -> ModelConfig:
        """Load ``configs/model/<name>.yaml``."""
        path = PRESET_DIR / f"{name}.yaml"
        if not path.exists():
            available = ", ".join(sorted(p.stem for p in PRESET_DIR.glob("*.yaml")))
            raise ConfigError(f"unknown model preset {name!r} (available: {available})")
        return cls.from_yaml(path)


def param_count(cfg: ModelConfig) -> dict[str, int]:
    """Exact parameter count by component (no biases anywhere)."""
    d, hd = cfg.d_model, cfg.head_dim
    attn = d * cfg.n_heads * hd * 2 + d * cfg.n_kv_heads * hd * 2  # wq, wo, wk, wv
    mlp = 3 * d * cfg.ffn_hidden  # gate, up, down
    norms = 2 * d  # pre-attention + pre-MLP RMSNorm weights
    blocks = cfg.n_layers * (attn + mlp + norms)
    embedding = cfg.vocab_size * d
    lm_head = 0 if cfg.tie_embeddings else cfg.vocab_size * d
    final_norm = d
    total = embedding + blocks + final_norm + lm_head
    return {
        "embedding": embedding,
        "blocks": blocks,
        "final_norm": final_norm,
        "lm_head_untied": lm_head,
        "non_embedding": blocks + final_norm,
        "total": total,
    }


def flops_per_token(cfg: ModelConfig, seq_len: int | None = None) -> dict[str, float]:
    """Estimated FLOPs per token for a forward pass and a full training step.

    Uses the standard accounting (Kaplan et al. 2020; PaLM appendix B):
      * every weight in a matmul costs 2 FLOPs/token forward (multiply + add).
        That's all parameters except norms, and the embedding *lookup* is free.
        With tied weights the shared matrix is still used once, as the LM head;
      * attention scores (QK^T) and the weighted sum (AV) cost
        2 * 2 * n_layers * seq_len * d_model per token forward without masking;
        a causal mask halves the useful work, so we count half of that;
      * backward costs ~2x forward, so training = 3x forward.
    """
    t = seq_len or cfg.context_length
    p = param_count(cfg)
    matmul_params = p["blocks"] - cfg.n_layers * 2 * cfg.d_model + cfg.vocab_size * cfg.d_model
    attention = 2 * cfg.n_layers * t * cfg.d_model  # causal: half of 4 * L * T * d
    forward = 2 * matmul_params + attention
    return {"forward": float(forward), "train": float(3 * forward)}
