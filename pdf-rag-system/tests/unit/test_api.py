"""HTTP API: validation, response structure, and error mapping (all dependencies faked)."""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from app.application import create_app
from app.container import ServiceContainer
from app.core.exceptions import DatabaseConnectionError, LLMError, LLMRefusalError
from app.embeddings.embedding_service import EmbeddingService
from app.llm.prompts import NOT_FOUND_ANSWER
from app.retrieval.context_builder import ContextBuilder
from app.retrieval.retriever import Retriever
from app.services.rag_service import RAGService
from app.vectorstore.models import DatabaseHealth, NewChunk, NewDocument
from tests.conftest import NO_WAIT
from tests.helpers import FakeEmbeddingProvider, FakeHealth, FakeLLM, InMemoryRepository, WhitespaceTokenizer, bag_of_words_vector

DIMS = 512
ARCH_TEXT = "The system architecture has an ingestion server and a retrieval server sharing PostgreSQL."


def seeded_repository() -> InMemoryRepository:
    repo = InMemoryRepository()
    repo.insert_document(
        NewDocument(filename="architecture.pdf", file_hash="a" * 64, file_size=10, total_pages=12, embedding_model="fake", chunk_size=800, chunk_overlap=150),
        [
            NewChunk(0, ARCH_TEXT, 12, 12, 14, bag_of_words_vector(ARCH_TEXT, DIMS)),
            NewChunk(1, "Unrelated appendix about office plants.", 13, 13, 6, bag_of_words_vector("Unrelated appendix about office plants.", DIMS)),
        ],
    )
    return repo


def build_client(*, llm: FakeLLM | None = None, repo: InMemoryRepository | None = None, health: FakeHealth | None = None, make_settings=None, answer_format: str = "plain"):
    repo = repo or seeded_repository()
    llm = llm or FakeLLM()
    retriever = Retriever(
        embedding_service=EmbeddingService(FakeEmbeddingProvider(dims=DIMS), dimensions=DIMS, retry_policy=NO_WAIT),
        searcher=repo,
        default_top_k=5,
        similarity_threshold=0.3,
    )
    service = RAGService(retriever=retriever, context_builder=ContextBuilder(WhitespaceTokenizer(), max_tokens=2000), llm=llm, answer_format=answer_format)
    settings = make_settings()
    container = ServiceContainer(settings=settings, database=health or FakeHealth(), documents=repo, rag_service=service)
    app = create_app(settings, container_factory=lambda _: container, configure_logs=False)
    return TestClient(app), llm, repo


@pytest.fixture
def client_factory(make_settings):
    def factory(**kw):
        return build_client(make_settings=make_settings, **kw)
    return factory


# --------------------------------------------------------------------------
# /api/v1/rag/query
# --------------------------------------------------------------------------


def test_query_returns_answer_with_sources(client_factory) -> None:
    client, llm, _ = client_factory()
    with client:
        resp = client.post("/api/v1/rag/query", json={"question": "Explain the system architecture", "top_k": 5})

    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"success", "question", "answer", "sources", "model", "provider", "usage"}
    assert body["success"] is True
    assert body["question"] == "Explain the system architecture"
    assert body["answer"] == llm.answer
    assert body["provider"] == "claude" and body["model"] == "fake-model"
    assert body["usage"] == {"input_tokens": 1200, "output_tokens": 250}

    source = body["sources"][0]
    assert {"document", "page", "chunk_id", "relevance_score", "content"} <= set(source)
    assert source["document"] == "architecture.pdf" and source["page"] == 12
    assert 0 < source["relevance_score"] <= 1
    uuid.UUID(source["chunk_id"])
    assert all(s["content"] != "Unrelated appendix about office plants." for s in body["sources"])

    # The LLM saw the retrieved context and the question.
    system_prompt, user_prompt = llm.prompts[0]
    assert ARCH_TEXT in user_prompt and "Explain the system architecture" in user_prompt
    assert 'document="architecture.pdf" pages="p. 12"' in user_prompt


def test_response_schema_is_identical_for_both_providers(client_factory) -> None:
    shapes = []
    for name in ("claude", "openai"):
        client, _, _ = client_factory(llm=FakeLLM(name=name, model=f"{name}-model"))
        with client:
            body = client.post("/api/v1/rag/query", json={"question": "system architecture"}).json()
        shapes.append((set(body), set(body["sources"][0]), set(body["usage"])))
        assert body["provider"] == name
    assert shapes[0] == shapes[1]


def test_no_relevant_context_returns_not_found_without_calling_llm(client_factory) -> None:
    client, llm, _ = client_factory()
    with client:
        resp = client.post("/api/v1/rag/query", json={"question": "zebra migration patterns in savanna"})
    body = resp.json()
    assert resp.status_code == 200
    assert body["answer"] == NOT_FOUND_ANSWER
    assert body["sources"] == [] and body["usage"] == {"input_tokens": 0, "output_tokens": 0}
    assert llm.prompts == []


def test_filters_restrict_documents(client_factory) -> None:
    client, _, _ = client_factory()
    with client:
        body = client.post(
            "/api/v1/rag/query",
            json={"question": "system architecture", "filters": {"filenames": ["other.pdf"]}},
        ).json()
    assert body["sources"] == [] and body["answer"] == NOT_FOUND_ANSWER


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": ""},
        {"question": "   "},
        {"question": "ok", "top_k": 0},
        {"question": "ok", "top_k": 51},
        {"question": "ok", "top_k": "five"},
        {"question": "ok", "similarity_threshold": 1.5},
        {"question": "ok", "unknown": True},
        {"question": "ok", "filters": {"page_from": 5, "page_to": 2}},
        {"question": "x" * 4001},
    ],
)
def test_query_validation_errors(client_factory, payload) -> None:
    client, llm, _ = client_factory()
    with client:
        resp = client.post("/api/v1/rag/query", json=payload)
    assert resp.status_code == 422
    body = resp.json()
    assert body["success"] is False
    assert body["error"]["code"] == "validation_error"
    assert isinstance(body["error"]["details"], list) and body["error"]["details"]
    assert llm.prompts == []


def test_malformed_json_is_422(client_factory) -> None:
    client, _, _ = client_factory()
    with client:
        resp = client.post("/api/v1/rag/query", content=b"{not json", headers={"content-type": "application/json"})
    assert resp.status_code == 422 and resp.json()["error"]["code"] == "validation_error"


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (LLMError("Anthropic rate limit exceeded", retryable=True), 502, "llm_error"),
        (LLMRefusalError("The model declined to answer this request"), 422, "llm_refusal"),
        (DatabaseConnectionError("Lost connection to PostgreSQL"), 503, "database_unavailable"),
    ],
)
def test_upstream_errors_map_to_structured_responses(client_factory, error, status, code) -> None:
    client, _, _ = client_factory(llm=FakeLLM(error=error))
    with client:
        resp = client.post("/api/v1/rag/query", json={"question": "system architecture"})
    assert resp.status_code == status
    body = resp.json()
    assert body["success"] is False and body["error"]["code"] == code
    assert body["request_id"] == resp.headers["x-request-id"]


def test_unexpected_errors_do_not_leak_details(client_factory) -> None:
    client, _, _ = client_factory(llm=FakeLLM(error=RuntimeError("secret internal detail sk-ant-abcdefghijk")))
    with client:
        resp = client.post("/api/v1/rag/query", json={"question": "system architecture"})
    assert resp.status_code == 500
    assert resp.json()["error"] == {"code": "internal_error", "message": "Internal server error."}


# --------------------------------------------------------------------------
# /health and /api/v1/documents
# --------------------------------------------------------------------------


def test_health_ok(client_factory) -> None:
    client, _, _ = client_factory()
    with client:
        resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok" and body["api"] == "ok"
    assert body["database"]["status"] == "ok" and body["database"]["pgvector_version"] == "0.8.0"


def test_health_reports_database_down(client_factory) -> None:
    client, _, _ = client_factory(health=FakeHealth(DatabaseHealth(ok=False, error="Timed out connecting to PostgreSQL")))
    with client:
        resp = client.get("/health")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "degraded" and body["database"]["status"] == "error"


def test_list_documents(client_factory) -> None:
    client, _, _ = client_factory()
    with client:
        resp = client.get("/api/v1/documents")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True and body["total"] == 1
    doc = body["documents"][0]
    assert doc["filename"] == "architecture.pdf" and doc["total_pages"] == 12 and doc["total_chunks"] == 2


def test_documents_pagination_validation(client_factory) -> None:
    client, _, _ = client_factory()
    with client:
        assert client.get("/api/v1/documents?limit=0").status_code == 422
        assert client.get("/api/v1/documents?limit=1&offset=5").json()["documents"] == []


def test_unknown_route_is_structured_404(client_factory) -> None:
    client, _, _ = client_factory()
    with client:
        resp = client.get("/api/v1/nope")
    assert resp.status_code == 404 and resp.json()["error"]["code"] == "not_found"


def test_request_id_is_propagated(client_factory) -> None:
    client, _, _ = client_factory()
    with client:
        resp = client.get("/health", headers={"X-Request-ID": "abc-123"})
    assert resp.headers["x-request-id"] == "abc-123"


# --------------------------------------------------------------------------
# Spoken answers (text-to-speech)
# --------------------------------------------------------------------------

SPOKEN_LLM_OUTPUT = (
    "Sure, I'd be happy to help! The system has an ingestion server and a retrieval server [1].\n\n"
    "Shared storage:\n- Both use PostgreSQL [1]\n- Vectors live in pgvector"
)


def test_spoken_answer_is_clean_for_text_to_speech(client_factory) -> None:
    client, llm, _ = client_factory(llm=FakeLLM(answer=SPOKEN_LLM_OUTPUT), answer_format="spoken")
    with client:
        body = client.post("/api/v1/rag/query", json={"question": "Explain the system architecture"}).json()

    assert body["success"] is True
    assert body["answer"] == (
        "The system has an ingestion server and a retrieval server.\n\n"
        "Both use PostgreSQL. Vectors live in pgvector."
    )
    assert body["sources"], "sources are still returned for the app, just not spoken"

    system_prompt, user_prompt = llm.prompts[0]
    assert "text-to-speech" in system_prompt and "Never write anything in square brackets" in system_prompt
    assert "Reply to the question as natural speech" in user_prompt


def test_spoken_not_found_answer(client_factory) -> None:
    from app.llm.prompts import SPOKEN_NOT_FOUND_ANSWER

    client, llm, _ = client_factory(answer_format="spoken")
    with client:
        body = client.post("/api/v1/rag/query", json={"question": "zebra migration patterns in savanna"}).json()
    assert body["answer"] == SPOKEN_NOT_FOUND_ANSWER and llm.prompts == []
