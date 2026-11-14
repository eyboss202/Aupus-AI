"""Vulpes AI: a small English language model built completely from scratch.

Sub-packages:
    tokenizer  byte-level BPE (training, encoding, decoding, special tokens)
    data       download, cleaning, dedup, tokenization into shards, dataloader
    model      model config, layers, transformer, generation
    train      pretraining + SFT loops, optimizer, scheduler, checkpointing
    eval       perplexity and zero-shot benchmark harness
    cli        chat / generate entry points
"""

import os

__version__ = "0.1.0.dev0"

# PyTorch's CUDA allocator caches freed blocks to reuse them. With batches of varying
# shape (SFT conversations, evaluation) it keeps reserving new blocks, and on Linux it
# empties the cache when the GPU runs out. On Windows the driver never runs out: it quietly
# moves the excess into shared system RAM, and training gets ~4x slower (docs/PLAN.md §19,
# measured: 1 GB in use, 10.75 GB reserved). Expandable segments grow and shrink one
# mapping instead, so reserved memory stays close to what is in use. It must be set before
# PyTorch first allocates CUDA memory, so it is set here, when any part of Vulpes is
# imported; an explicit PYTORCH_CUDA_ALLOC_CONF in the environment wins.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
