"""OpenAI embeddings (text-embedding-3-small by default)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import openai

from app.core.exceptions import ConfigurationError, EmbeddingError
from app.core.http_errors import is_retryable_status, retry_after_seconds
from app.embeddings.base import EmbeddingBatch, EmbeddingProvider


class OpenAIEmbeddingProvider(EmbeddingProvider):
    name = "openai"

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        dimensions: int | None = None,
        timeout: float = 60.0,
        client: Any | None = None,
    ) -> None:
        if client is None and not api_key:
            raise ConfigurationError("OPENAI_API_KEY is required for OpenAI embeddings")
        self.model = model
        self._dimensions = dimensions
        # SDK retries are disabled: EmbeddingService owns retry/backoff.
        self._client = client or openai.OpenAI(api_key=api_key, max_retries=0, timeout=timeout)

    def embed_batch(self, texts: Sequence[str]) -> EmbeddingBatch:
        kwargs: dict[str, Any] = {"model": self.model, "input": list(texts), "encoding_format": "float"}
        # Only text-embedding-3* models accept a custom output dimension.
        if self._dimensions and self.model.startswith("text-embedding-3"):
            kwargs["dimensions"] = self._dimensions
        try:
            response = self._client.embeddings.create(**kwargs)
        except openai.RateLimitError as exc:
            # insufficient_quota is a 429 that will never succeed on retry.
            quota = getattr(exc, "code", None) == "insufficient_quota"
            raise EmbeddingError(
                "OpenAI quota exhausted" if quota else "OpenAI embeddings rate limit exceeded",
                retryable=not quota,
                retry_after=retry_after_seconds(getattr(exc, "response", None)),
            ) from exc
        except openai.APITimeoutError as exc:
            raise EmbeddingError("OpenAI embeddings request timed out", retryable=True) from exc
        except openai.APIConnectionError as exc:
            raise EmbeddingError("Could not reach the OpenAI API", retryable=True) from exc
        except openai.AuthenticationError as exc:
            raise EmbeddingError("OpenAI rejected the API key (authentication failed)") from exc
        except openai.APIStatusError as exc:
            raise EmbeddingError(
                f"OpenAI embeddings request failed with HTTP {exc.status_code}",
                retryable=is_retryable_status(exc.status_code),
                retry_after=retry_after_seconds(exc.response),
                details={"status_code": exc.status_code},
            ) from exc

        data = sorted(response.data, key=lambda item: item.index)
        usage = getattr(response, "usage", None)
        return EmbeddingBatch(
            vectors=[list(item.embedding) for item in data],
            total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
        )
