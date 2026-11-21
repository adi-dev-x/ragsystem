"""Embedding provider interface.

Providers make exactly one API request per call and translate SDK exceptions
into :class:`~app.core.exceptions.EmbeddingError` (with ``retryable`` set for
transient failures). Batching, retries and validation live in
:class:`~app.embeddings.embedding_service.EmbeddingService`, so a new provider
only has to implement :meth:`embed_batch`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EmbeddingBatch:
    vectors: list[list[float]]
    total_tokens: int


class EmbeddingProvider(ABC):
    name: str
    model: str

    @abstractmethod
    def embed_batch(self, texts: Sequence[str]) -> EmbeddingBatch:
        """Embed ``texts`` in a single request, preserving input order."""
