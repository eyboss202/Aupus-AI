"""Aupus AI: a small English language model built completely from scratch.

Sub-packages:
    tokenizer  byte-level BPE (training, encoding, decoding, special tokens)
    data       download, cleaning, dedup, tokenization into shards, dataloader
    model      model config, layers, transformer, generation
    train      pretraining + SFT loops, optimizer, scheduler, checkpointing
    eval       perplexity and zero-shot benchmark harness
    cli        chat / generate entry points
"""

__version__ = "0.1.0.dev0"
