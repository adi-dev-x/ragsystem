"""Token counting shared by the chunker and the context builder."""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol, Sequence

import tiktoken


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...

    def decode(self, tokens: Sequence[int]) -> str: ...

    def count(self, text: str) -> int: ...


class TiktokenTokenizer:
    """tiktoken wrapper. ``cl100k_base`` is the encoding used by text-embedding-3-*."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self._encoding = _load_encoding(encoding_name)

    def encode(self, text: str) -> list[int]:
        return self._encoding.encode(text, disallowed_special=())

    def decode(self, tokens: Sequence[int]) -> str:
        return self._encoding.decode(list(tokens))

    def count(self, text: str) -> int:
        return len(self.encode(text))


@lru_cache(maxsize=4)
def _load_encoding(name: str) -> tiktoken.Encoding:
    return tiktoken.get_encoding(name)
