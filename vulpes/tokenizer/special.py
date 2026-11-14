"""The special tokens of the Vulpes tokenizer.

Special tokens are control signals, not text. They get the last ids of the
vocabulary (after the 256 byte tokens and all learned merges), and BPE never
produces them. By default, ``Tokenizer.encode`` treats a literal ``"<|user|>"``
in input text as ordinary characters. Only code that builds prompts on purpose
(the chat template, the data pipeline) passes ``allow_special=True``. This stops
a user from typing ``<|assistant|>`` and taking over the conversation format.

With 7 special tokens and a 32,000-id vocabulary, BPE learns
32,000 - 256 - 7 = 31,737 merges.
"""

from __future__ import annotations

BOS = "<|bos|>"  # beginning of sequence (optional; reserved for later use)
EOS = "<|eos|>"  # end of document: separates documents in pretraining shards
PAD = "<|pad|>"  # padding in batches; its positions are always masked out of the loss
SYSTEM = "<|system|>"  # chat template: starts the system message
USER = "<|user|>"  # chat template: starts a user turn
ASSISTANT = "<|assistant|>"  # chat template: starts an assistant turn
END = "<|end|>"  # chat template: ends a turn; generation stops here

#: Order matters: it decides the ids (first special = first id after the merges).
SPECIAL_TOKENS: tuple[str, ...] = (BOS, EOS, PAD, SYSTEM, USER, ASSISTANT, END)
