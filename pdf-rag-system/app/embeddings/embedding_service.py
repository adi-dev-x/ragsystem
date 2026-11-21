"""Reusable embedding service: batching, retries, and dimension validation."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass

from app.config import Settings
from app.core.exceptions import ConfigurationError, EmbeddingDimensionError, EmbeddingError
from app.core.logging import get_logger
from app.core.retry import RetryPolicy, retry_call
from app.embeddings.base import EmbeddingProvider

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    vectors: list[list[float]]
    total_tokens: int
    requests: int


class EmbeddingService:
    def __init__(
        self,
        provider: EmbeddingProvider,
        *,
        dimensions: int,
        batch_size: int = 100,
        retry_policy: RetryPolicy | None = None,
        document_prefix: str = "",
        query_prefix: str = "",
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.provider = provider
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.retry_policy = retry_policy or RetryPolicy()
        self.document_prefix = document_prefix
        self.query_prefix = query_prefix

    @property
    def model(self) -> str:
        return self.provider.model

    def embed_texts(self, texts: Sequence[str]) -> EmbeddingResult:
        """Embed document passages in batches; output order matches input order."""
        return self._embed(texts, self.document_prefix)

    def embed_query(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise ValueError("Query text must not be empty")
        return self._embed([text], self.query_prefix).vectors[0]

    def _embed(self, texts: Sequence[str], prefix: str) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult(vectors=[], total_tokens=0, requests=0)
        blank = [i for i, t in enumerate(texts) if not t or not t.strip()]
        if blank:
            # Empty strings are rejected by the API and waste a request.
            raise ValueError(f"Cannot embed empty text (indexes {blank[:10]})")

        vectors: list[list[float]] = []
        total_tokens = 0
        batches = range(0, len(texts), self.batch_size)
        for number, start in enumerate(batches, start=1):
            batch = [prefix + t for t in texts[start: start + self.batch_size]]
            started = time.perf_counter()
            result = retry_call(
                lambda batch=batch: self.provider.embed_batch(batch),
                policy=self.retry_policy,
                operation=f"embeddings:{self.provider.name}",
            )
            self._validate(result.vectors, expected=len(batch))
            vectors.extend(result.vectors)
            total_tokens += result.total_tokens
            logger.debug(
                "embedding batch complete",
                extra={
                    "batch": number,
                    "batches": len(batches),
                    "inputs": len(batch),
                    "tokens": result.total_tokens,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                },
            )
        return EmbeddingResult(vectors=vectors, total_tokens=total_tokens, requests=len(batches))

    def _validate(self, vectors: list[list[float]], *, expected: int) -> None:
        if len(vectors) != expected:
            raise EmbeddingError(f"Embedding provider returned {len(vectors)} vectors for {expected} inputs")
        for i, vector in enumerate(vectors):
            if len(vector) != self.dimensions:
                raise EmbeddingDimensionError(
                    f"Embedding has {len(vector)} dimensions, expected {self.dimensions}",
                    details={"index": i, "model": self.model},
                )
            if not all(math.isfinite(v) for v in vector):
                raise EmbeddingError("Embedding contains NaN or infinite values", details={"index": i})


def create_embedding_service(settings: Settings) -> EmbeddingService:
    """Build the configured embedding service (independent of LLM_PROVIDER)."""
    provider: EmbeddingProvider
    if settings.embedding_provider == "openai":
        from app.embeddings.openai_provider import OpenAIEmbeddingProvider

        provider = OpenAIEmbeddingProvider(
            api_key=settings.openai_api_key.get_secret_value() if settings.openai_api_key else None,
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
            timeout=settings.embedding_timeout_seconds,
        )
    elif settings.embedding_provider == "ollama":
        from app.core.ollama import OllamaClient
        from app.embeddings.ollama_provider import OllamaEmbeddingProvider

        provider = OllamaEmbeddingProvider(
            client=OllamaClient(settings.ollama_base_url, timeout=settings.embedding_timeout_seconds),
            model=settings.embedding_model,
            keep_alive=settings.ollama_keep_alive,
        )
    else:  # pragma: no cover - guarded by the Settings Literal type
        raise ConfigurationError(f"Unsupported EMBEDDING_PROVIDER: {settings.embedding_provider}")

    return EmbeddingService(
        provider,
        dimensions=settings.embedding_dimensions,
        batch_size=settings.embedding_batch_size,
        retry_policy=RetryPolicy(
            max_attempts=settings.retry_max_attempts,
            base_delay=settings.retry_base_delay_seconds,
            max_delay=settings.retry_max_delay_seconds,
        ),
        document_prefix=settings.embedding_document_prefix,
        query_prefix=settings.embedding_query_prefix,
    )
