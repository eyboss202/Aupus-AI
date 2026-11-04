"""Llama-style decoder-only transformer: config, layers, model and generation (Milestone 4)."""

from aupus.model.config import ModelConfig, flops_per_token, param_count

__all__ = ["ModelConfig", "flops_per_token", "param_count"]
