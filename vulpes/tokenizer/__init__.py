"""Byte-level BPE tokenizer written from scratch: train, encode, decode, special tokens.

Typical use::

    from vulpes.tokenizer import EOS, Tokenizer

    tok = Tokenizer.load("data/tokenizer/vulpes-bpe-32k.json")
    ids = tok.encode("Hello, world!") + [tok.special_id(EOS)]
    text = tok.decode(ids, skip_special=True)
"""

from vulpes.tokenizer.pretokenize import PATTERN, pretokenize
from vulpes.tokenizer.special import ASSISTANT, BOS, END, EOS, PAD, SPECIAL_TOKENS, SYSTEM, USER
from vulpes.tokenizer.tokenizer import Tokenizer, TokenizerError
from vulpes.tokenizer.train import count_chunks, train_bpe

__all__ = [
    "ASSISTANT",
    "BOS",
    "END",
    "EOS",
    "PAD",
    "PATTERN",
    "SPECIAL_TOKENS",
    "SYSTEM",
    "USER",
    "Tokenizer",
    "TokenizerError",
    "count_chunks",
    "pretokenize",
    "train_bpe",
]
