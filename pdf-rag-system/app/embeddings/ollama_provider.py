"""Embeddings from a local Ollama model (e.g. nomic-embed-text, mxbai-embed-large)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from app.core.exceptions import EmbeddingError
from app.core.ollama import OllamaClient
from app.embeddings.base import EmbeddingBatch, EmbeddingProvider


class OllamaEmbeddingProvider(EmbeddingProvider):
    name = "ollama"

    def __init__(self, *, client: OllamaClient, model: str, keep_alive: str | None = None) -> None:
        self.model = model
        self._client = client
        self._keep_alive = keep_alive

    def embed_batch(self, texts: Sequence[str]) -> EmbeddingBatch:
        payload: dict[str, Any] = {"model": self.model, "input": list(texts)}
        if self._keep_alive:
            payload["keep_alive"] = self._keep_alive
        data = self._client.post("/api/embed", payload, error_cls=EmbeddingError)

        vectors = data.get("embeddings")
        if not isinstance(vectors, list):
            raise EmbeddingError("Ollama response has no 'embeddings' field", retryable=True)
        return EmbeddingBatch(
            vectors=[[float(x) for x in v] for v in vectors],
            total_tokens=int(data.get("prompt_eval_count") or 0),
        )
