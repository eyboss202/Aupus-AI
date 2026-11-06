"""Llama-style decoder-only transformer: config, layers, model and generation."""

from aupus.model.config import ModelConfig, flops_per_token, param_count
from aupus.model.generate import generate, sample_next, stream
from aupus.model.layers import KVCache
from aupus.model.transformer import IGNORE_INDEX, Output, Transformer

__all__ = [
    "IGNORE_INDEX",
    "KVCache",
    "ModelConfig",
    "Output",
    "Transformer",
    "flops_per_token",
    "generate",
    "param_count",
    "sample_next",
    "stream",
]
