from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx2
import openai
import pytest

from app.core.exceptions import ConfigurationError, EmbeddingDimensionError, EmbeddingError
from app.embeddings.embedding_service import EmbeddingService, create_embedding_service
from app.embeddings.openai_provider import OpenAIEmbeddingProvider
from tests.conftest import NO_WAIT
from tests.helpers import FakeEmbeddingProvider


def _response(status: int, headers: dict[str, str] | None = None) -> httpx2.Response:
    return httpx2.Response(status, request=httpx2.Request("POST", "https://api.openai.com/v1/embeddings"), headers=headers)


def _embedding_response(vectors: list[list[float]], tokens: int = 7) -> SimpleNamespace:
    # API may return items out of order; the provider sorts by index.
    data = [SimpleNamespace(index=i, embedding=v) for i, v in enumerate(vectors)][::-1]
    return SimpleNamespace(data=data, usage=SimpleNamespace(total_tokens=tokens))


# --------------------------------------------------------------------------
# EmbeddingService
# --------------------------------------------------------------------------


def test_batches_requests_and_preserves_order() -> None:
    provider = FakeEmbeddingProvider(dims=16)
    service = EmbeddingService(provider, dimensions=16, batch_size=100, retry_policy=NO_WAIT)
    texts = [f"text number {i}" for i in range(250)]

    result = service.embed_texts(texts)

    assert [len(c) for c in provider.calls] == [100, 100, 50]
    assert result.requests == 3 and len(result.vectors) == 250
    single = FakeEmbeddingProvider(dims=16).embed_batch([texts[137]]).vectors[0]
    assert result.vectors[137] == single


def test_retries_transient_errors_then_succeeds() -> None:
    provider = FakeEmbeddingProvider(dims=16, fail_times=2)
    service = EmbeddingService(provider, dimensions=16, retry_policy=NO_WAIT)
    assert len(service.embed_texts(["hello"]).vectors) == 1
    assert len(provider.calls) == 3


def test_gives_up_after_max_attempts() -> None:
    provider = FakeEmbeddingProvider(dims=16, fail_times=10)
    service = EmbeddingService(provider, dimensions=16, retry_policy=NO_WAIT)
    with pytest.raises(EmbeddingError):
        service.embed_texts(["hello"])
    assert len(provider.calls) == NO_WAIT.max_attempts


def test_permanent_errors_are_not_retried() -> None:
    provider = FakeEmbeddingProvider(dims=16)
    provider.fail_on = "bad"
    service = EmbeddingService(provider, dimensions=16, retry_policy=NO_WAIT)
    with pytest.raises(EmbeddingError):
        service.embed_texts(["bad input"])
    assert len(provider.calls) == 1


def test_dimension_mismatch_is_rejected() -> None:
    service = EmbeddingService(FakeEmbeddingProvider(dims=8), dimensions=1536, retry_policy=NO_WAIT)
    with pytest.raises(EmbeddingDimensionError):
        service.embed_texts(["hello"])


def test_empty_texts_are_rejected_without_api_calls() -> None:
    provider = FakeEmbeddingProvider(dims=16)
    service = EmbeddingService(provider, dimensions=16, retry_policy=NO_WAIT)
    with pytest.raises(ValueError):
        service.embed_texts(["ok", "   "])
    with pytest.raises(ValueError):
        service.embed_query("")
    assert service.embed_texts([]).vectors == []
    assert provider.calls == []


# --------------------------------------------------------------------------
# OpenAI provider (mocked client)
# --------------------------------------------------------------------------


def test_openai_provider_request_and_response_mapping() -> None:
    client = MagicMock()
    client.embeddings.create.return_value = _embedding_response([[0.1, 0.2], [0.3, 0.4]], tokens=11)
    provider = OpenAIEmbeddingProvider(api_key=None, model="text-embedding-3-small", dimensions=2, client=client)

    batch = provider.embed_batch(["a", "b"])

    assert batch.vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert batch.total_tokens == 11
    kwargs = client.embeddings.create.call_args.kwargs
    assert kwargs["model"] == "text-embedding-3-small" and kwargs["input"] == ["a", "b"] and kwargs["dimensions"] == 2


def test_openai_rate_limit_is_retryable_with_retry_after() -> None:
    client = MagicMock()
    client.embeddings.create.side_effect = openai.RateLimitError(
        "slow down", response=_response(429, {"retry-after": "2"}), body=None
    )
    provider = OpenAIEmbeddingProvider(api_key=None, model="text-embedding-3-small", client=client)
    with pytest.raises(EmbeddingError) as info:
        provider.embed_batch(["a"])
    assert info.value.retryable and info.value.retry_after == 2.0


def test_openai_insufficient_quota_is_not_retryable() -> None:
    client = MagicMock()
    client.embeddings.create.side_effect = openai.RateLimitError(
        "quota", response=_response(429), body={"code": "insufficient_quota"}
    )
    provider = OpenAIEmbeddingProvider(api_key=None, model="text-embedding-3-small", client=client)
    with pytest.raises(EmbeddingError) as info:
        provider.embed_batch(["a"])
    assert not info.value.retryable


def test_openai_auth_error_is_not_retryable_and_hides_key() -> None:
    client = MagicMock()
    client.embeddings.create.side_effect = openai.AuthenticationError(
        "Incorrect API key provided: sk-abc", response=_response(401), body=None
    )
    provider = OpenAIEmbeddingProvider(api_key=None, model="text-embedding-3-small", client=client)
    with pytest.raises(EmbeddingError) as info:
        provider.embed_batch(["a"])
    assert not info.value.retryable
    assert "sk-" not in info.value.message


def test_openai_server_error_is_retryable() -> None:
    client = MagicMock()
    client.embeddings.create.side_effect = openai.InternalServerError("boom", response=_response(503), body=None)
    provider = OpenAIEmbeddingProvider(api_key=None, model="text-embedding-3-small", client=client)
    with pytest.raises(EmbeddingError) as info:
        provider.embed_batch(["a"])
    assert info.value.retryable


def test_factory_requires_openai_key(make_settings) -> None:
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        create_embedding_service(make_settings(openai_api_key=None))
    service = create_embedding_service(make_settings())
    assert service.model == "text-embedding-3-small" and service.dimensions == 1536
