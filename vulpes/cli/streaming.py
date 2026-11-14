"""Print generated text as it is produced, without breaking multi-byte characters.

A token is a sequence of *bytes*, not characters. "é" is two bytes in UTF-8 and
an emoji is four, and BPE may put them in different tokens. Decoding each token
on its own would print replacement characters (``�``) for the halves. So bytes
go through Python's incremental UTF-8 decoder: it returns whatever complete
characters it has and keeps an unfinished sequence until the next token
completes it.
"""

from __future__ import annotations

import codecs

from vulpes.tokenizer import Tokenizer


class TextStream:
    """Turn token ids into text pieces as they arrive. Special tokens produce no text."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.text = ""

    def push(self, token_id: int) -> str:
        """Add one token; return the newly completed text (possibly empty)."""
        if self.tokenizer.is_special(token_id):
            return ""
        piece = self._decoder.decode(self.tokenizer.token_bytes(token_id))
        self.text += piece
        return piece

    def flush(self) -> str:
        """End of generation: return any leftover (an incomplete character becomes ``�``)."""
        piece = self._decoder.decode(b"", final=True)
        self.text += piece
        return piece
