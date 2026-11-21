"""Ollama embedding and chat providers against a mocked HTTP transport."""

from __future__ import annotations

import json

import httpx
import pytest

from app.core.exceptions import ConfigurationError, EmbeddingError, LLMError
from app.core.ollama import OllamaClient
from app.core.retry import RetryPolicy
from app.embeddings.embedding_service import EmbeddingService, create_embedding_service
from app.embeddings.ollama_provider import OllamaEmbeddingProvider
from app.llm.factory import create_llm_provider
from app.llm.ollama_provider import OllamaProvider

NO_WAIT = RetryPolicy(max_attempts=3, base_delay=0, max_delay=0, jitter=False)


class Recorder:
    """httpx transport that records requests and replies with a canned handler."""

    def __init__(self, handler):
        self.requests: list[tuple[str, dict]] = []
        self._handler = handler

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        self.requests.append((request.url.path, body))
        return self._handler(request, body)


def client_for(handler) -> tuple[OllamaClient, Recorder]:
    recorder = Recorder(handler)
    return OllamaClient("http://ollama.test:11434/", timeout=5, transport=httpx.MockTransport(recorder)), recorder


def embed_ok(request: httpx.Request, body: dict) -> httpx.Response:
    return httpx.Response(200, json={
        "model": body["model"],
        "embeddings": [[float(len(t)), 1.0, 0.0] for t in body["input"]],
        "prompt_eval_count": 42,
    })


# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------


def test_embedding_request_and_response() -> None:
    client, rec = client_for(embed_ok)
    batch = OllamaEmbeddingProvider(client=client, model="nomic-embed-text", keep_alive="10m").embed_batch(["ab", "abcd"])

    assert batch.vectors == [[2.0, 1.0, 0.0], [4.0, 1.0, 0.0]]
    assert batch.total_tokens == 42
    path, body = rec.requests[0]
    assert path == "/api/embed"
    assert body == {"model": "nomic-embed-text", "input": ["ab", "abcd"], "keep_alive": "10m"}


def test_document_and_query_prefixes() -> None:
    client, rec = client_for(embed_ok)
    service = EmbeddingService(
        OllamaEmbeddingProvider(client=client, model="nomic-embed-text"),
        dimensions=3,
        retry_policy=NO_WAIT,
        document_prefix="search_document: ",
        query_prefix="search_query: ",
    )
    service.embed_texts(["passage"])
    service.embed_query("question?")
    assert rec.requests[0][1]["input"] == ["search_document: passage"]
    assert rec.requests[1][1]["input"] == ["search_query: question?"]


def test_missing_model_gives_pull_hint() -> None:
    client, _ = client_for(lambda r, b: httpx.Response(404, json={"error": f'model "{b["model"]}" not found, try pulling it first'}))
    with pytest.raises(EmbeddingError, match="ollama pull nomic-embed-text") as info:
        OllamaEmbeddingProvider(client=client, model="nomic-embed-text").embed_batch(["x"])
    assert not info.value.retryable


def test_server_not_running_is_not_retried() -> None:
    def refuse(request, body):
        raise httpx.ConnectError("connection refused", request=request)

    client, rec = client_for(refuse)
    service = EmbeddingService(OllamaEmbeddingProvider(client=client, model="m"), dimensions=3, retry_policy=NO_WAIT)
    with pytest.raises(EmbeddingError, match="ollama serve"):
        service.embed_texts(["x"])
    assert len(rec.requests) == 1


def test_server_errors_are_retried() -> None:
    replies = iter([httpx.Response(500, json={"error": "loading model"}), None])

    def flaky(request, body):
        reply = next(replies)
        return reply if reply is not None else embed_ok(request, body)

    client, rec = client_for(flaky)
    service = EmbeddingService(OllamaEmbeddingProvider(client=client, model="m"), dimensions=3, retry_policy=NO_WAIT)
    assert len(service.embed_texts(["x"]).vectors) == 1
    assert len(rec.requests) == 2


def test_embedding_factory_needs_no_openai_key(make_settings) -> None:
    service = create_embedding_service(
        make_settings(embedding_provider="ollama", embedding_model="nomic-embed-text", embedding_dimensions=768, openai_api_key=None)
    )
    assert service.provider.name == "ollama" and service.model == "nomic-embed-text" and service.dimensions == 768


# --------------------------------------------------------------------------
# Chat
# --------------------------------------------------------------------------


def chat_reply(content: str, **extra) -> httpx.Response:
    return httpx.Response(200, json={
        "model": "llama3.1:8b",
        "message": {"role": "assistant", "content": content},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 1500,
        "eval_count": 120,
        **extra,
    })


def make_chat(handler, **kw) -> tuple[OllamaProvider, Recorder]:
    client, rec = client_for(handler)
    return OllamaProvider(client=client, model="llama3.1:8b", max_output_tokens=2048, num_ctx=8192, retry_policy=NO_WAIT, **kw), rec


def test_chat_request_and_response() -> None:
    provider, rec = make_chat(lambda r, b: chat_reply("Two servers share PostgreSQL [a.pdf, p. 1]."))
    response = provider.generate(system_prompt="SYS", user_prompt="USER")

    assert response.text == "Two servers share PostgreSQL [a.pdf, p. 1]."
    assert (response.provider, response.model, response.input_tokens, response.output_tokens) == ("ollama", "llama3.1:8b", 1500, 120)
    path, body = rec.requests[0]
    assert path == "/api/chat"
    assert body["stream"] is False
    assert body["messages"] == [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"}]
    assert body["options"] == {"num_ctx": 8192, "num_predict": 2048}


def test_chat_temperature_only_when_configured() -> None:
    provider, rec = make_chat(lambda r, b: chat_reply("ok"), temperature=0.1)
    provider.generate(system_prompt="s", user_prompt="u")
    assert rec.requests[0][1]["options"]["temperature"] == 0.1


def test_reasoning_tags_are_removed() -> None:
    provider, _ = make_chat(lambda r, b: chat_reply("<think>\nlet me check the sources\n</think>\n\nThe answer [a.pdf, p. 2]."))
    assert provider.generate(system_prompt="s", user_prompt="u").text == "The answer [a.pdf, p. 2]."


def test_empty_answer_is_an_error() -> None:
    provider, _ = make_chat(lambda r, b: chat_reply("<think>only thinking</think>"))
    with pytest.raises(LLMError, match="empty"):
        provider.generate(system_prompt="s", user_prompt="u")


def test_chat_timeout_is_not_retried() -> None:
    def slow(request, body):
        raise httpx.ReadTimeout("timed out", request=request)

    recorder = Recorder(slow)
    client = OllamaClient("http://ollama.test", timeout=120, retry_on_timeout=False, transport=httpx.MockTransport(recorder))
    provider = OllamaProvider(client=client, model="llama3.1:8b", max_output_tokens=512, num_ctx=8192, retry_policy=NO_WAIT)
    with pytest.raises(LLMError, match="LLM_TIMEOUT_SECONDS") as info:
        provider.generate(system_prompt="s", user_prompt="u")
    assert not info.value.retryable
    assert len(recorder.requests) == 1


def test_embedding_timeout_is_retried() -> None:
    def slow(request, body):
        raise httpx.ReadTimeout("timed out", request=request)

    client, rec = client_for(slow)
    service = EmbeddingService(OllamaEmbeddingProvider(client=client, model="m"), dimensions=3, retry_policy=NO_WAIT)
    with pytest.raises(EmbeddingError):
        service.embed_texts(["x"])
    assert len(rec.requests) == NO_WAIT.max_attempts


def test_llm_factory_selects_ollama_without_api_keys(make_settings) -> None:
    provider = create_llm_provider(make_settings(llm_provider="ollama", ollama_model="llama3.1:8b", anthropic_api_key=None, openai_api_key=None))
    assert isinstance(provider, OllamaProvider) and provider.model == "llama3.1:8b"


def test_llm_factory_requires_ollama_model(make_settings) -> None:
    with pytest.raises(ConfigurationError, match="OLLAMA_MODEL"):
        create_llm_provider(make_settings(llm_provider="ollama", ollama_model=None))
