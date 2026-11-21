from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from app.config import Settings
from app.core.retry import RetryPolicy
from app.embeddings.embedding_service import EmbeddingService
from app.ingestion.chunker import TextChunker
from tests.helpers import FakeEmbeddingProvider, WhitespaceTokenizer

NO_WAIT = RetryPolicy(max_attempts=4, base_delay=0.0, max_delay=0.0, jitter=False)


@pytest.fixture
def make_settings() -> Callable[..., Settings]:
    """Settings built from explicit values only (ignores any local .env)."""

    def factory(**overrides: Any) -> Settings:
        values: dict[str, Any] = {
            "database_url": "postgresql://user:secret@localhost:5432/rag",
            "llm_provider": "claude",
            "anthropic_model": "claude-opus-5-5",
            "openai_model": "gpt-4.1-mini",
            "anthropic_api_key": "sk-ant-test-key-0000000000",
            "openai_api_key": "sk-test-key-000000000000000000",
            "log_format": "text",
        }
        values.update(overrides)
        return Settings(_env_file=None, **values)  # type: ignore[call-arg]

    return factory


@pytest.fixture
def tokenizer() -> WhitespaceTokenizer:
    return WhitespaceTokenizer()


@pytest.fixture
def fake_provider() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider(dims=64)


@pytest.fixture
def embedding_service(fake_provider: FakeEmbeddingProvider) -> EmbeddingService:
    return EmbeddingService(fake_provider, dimensions=64, batch_size=8, retry_policy=NO_WAIT)


@pytest.fixture
def chunker(tokenizer: WhitespaceTokenizer) -> TextChunker:
    return TextChunker(tokenizer, chunk_size=60, chunk_overlap=15)
