"""PostgreSQL + pgvector integration tests (run with TEST_DATABASE_URL set)."""

from __future__ import annotations

import math
from pathlib import Path

import psycopg
import pytest
from fastapi.testclient import TestClient

from app.application import create_app
from app.container import ServiceContainer
from app.core.exceptions import DatabaseError, IngestionLockError
from app.embeddings.embedding_service import EmbeddingService
from app.ingestion.chunker import TextChunker
from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.context_builder import ContextBuilder
from app.retrieval.retriever import Retriever
from app.services.rag_service import RAGService
from app.vectorstore.migrations import apply_migrations
from app.vectorstore.models import NewChunk, NewDocument, SearchFilters
from app.vectorstore.repository import DocumentRepository
from tests.conftest import NO_WAIT
from tests.helpers import FakeEmbeddingProvider, FakeLLM, WhitespaceTokenizer, bag_of_words_vector, make_pdf

pytestmark = pytest.mark.integration

DIMS = 1536


def unit_vector(*hot: int, dims: int = DIMS) -> list[float]:
    v = [0.0] * dims
    for i in hot:
        v[i] = 1.0
    norm = math.sqrt(len(hot))
    return [x / norm for x in v]


def new_doc(name: str, digest_char: str, pages: int = 3) -> NewDocument:
    return NewDocument(
        filename=name, file_hash=digest_char * 64, file_size=1000, total_pages=pages,
        embedding_model="fake-embedding", chunk_size=800, chunk_overlap=150, metadata={"title": name},
    )


def chunk(index: int, text: str, vector: list[float], page: int = 1, page_end: int | None = None) -> NewChunk:
    return NewChunk(index, text, page, page_end or page, len(text.split()), vector)


# --------------------------------------------------------------------------
# Schema & migrations
# --------------------------------------------------------------------------


def test_migrations_are_idempotent(database_url: str) -> None:
    assert apply_migrations(database_url) == []


def test_schema_has_pgvector_hnsw_index_and_dimension(database_url: str, repository: DocumentRepository) -> None:
    assert repository.get_embedding_dimension() == 1536
    with psycopg.connect(database_url) as conn:
        assert conn.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'").fetchone()
        indexdef = conn.execute(
            "SELECT indexdef FROM pg_indexes WHERE indexname = 'document_chunks_embedding_hnsw_idx'"
        ).fetchone()[0]
    assert "hnsw" in indexdef and "vector_cosine_ops" in indexdef


def test_similarity_query_can_use_hnsw_index(database: object, repository: DocumentRepository) -> None:
    repository.insert_document(new_doc("a.pdf", "a"), [chunk(0, "x", unit_vector(1))])
    vec = "[" + ",".join(str(v) for v in unit_vector(1)) + "]"
    with database.connection() as conn:  # type: ignore[attr-defined]
        conn.execute("SET LOCAL enable_seqscan = off")
        plan = "\n".join(
            r["QUERY PLAN"]
            for r in conn.execute(
                "EXPLAIN SELECT id FROM document_chunks ORDER BY embedding <=> %s::vector LIMIT 5", [vec]
            ).fetchall()
        )
    assert "document_chunks_embedding_hnsw_idx" in plan


# --------------------------------------------------------------------------
# Repository
# --------------------------------------------------------------------------


def test_insert_and_list_documents(repository: DocumentRepository) -> None:
    doc_id = repository.insert_document(
        new_doc("arch.pdf", "a"), [chunk(0, "alpha", unit_vector(1)), chunk(1, "beta", unit_vector(2), page=2)]
    )
    docs = repository.list_documents()
    assert [d.filename for d in docs] == ["arch.pdf"]
    assert docs[0].id == doc_id and docs[0].total_chunks == 2 and docs[0].metadata == {"title": "arch.pdf"}
    assert repository.count_chunks(doc_id) == 2
    assert repository.count_documents() == 1


def test_duplicate_hash_rejected(repository: DocumentRepository) -> None:
    repository.insert_document(new_doc("a.pdf", "a"), [])
    with pytest.raises(DatabaseError):
        repository.insert_document(new_doc("b.pdf", "a"), [])


def test_replace_is_transactional(repository: DocumentRepository) -> None:
    doc_id = repository.insert_document(new_doc("a.pdf", "a"), [chunk(0, "old content", unit_vector(1))])

    bad = [chunk(0, "new content", [0.1] * 10)]  # wrong dimension -> insert fails
    with pytest.raises(DatabaseError):
        repository.replace_document(doc_id, new_doc("a.pdf", "b"), bad)
    doc = repository.list_documents()[0]
    assert doc.file_hash == "a" * 64 and repository.count_chunks(doc_id) == 1  # rolled back

    repository.replace_document(doc_id, new_doc("a.pdf", "b"), [chunk(0, "new 1", unit_vector(3)), chunk(1, "new 2", unit_vector(4))])
    doc = repository.list_documents()[0]
    assert doc.file_hash == "b" * 64 and doc.total_chunks == 2 and doc.updated_at > doc.created_at
    assert repository.count_chunks(doc_id) == 2


def test_rename_swap_respects_unique_filenames(repository: DocumentRepository) -> None:
    a = repository.insert_document(new_doc("a.pdf", "a"), [])
    b = repository.insert_document(new_doc("b.pdf", "b"), [])
    repository.rename_documents([(a, "b.pdf"), (b, "a.pdf")])
    names = {d.id: d.filename for d in repository.list_documents()}
    assert names == {a: "b.pdf", b: "a.pdf"}


def test_delete_cascades_to_chunks(repository: DocumentRepository) -> None:
    doc_id = repository.insert_document(new_doc("a.pdf", "a"), [chunk(0, "x", unit_vector(1))])
    assert repository.delete_documents([doc_id]) == 1
    assert repository.count_chunks() == 0 and repository.count_documents() == 0


def test_vector_search_orders_by_cosine_similarity(repository: DocumentRepository) -> None:
    repository.insert_document(
        new_doc("a.pdf", "a"),
        [
            chunk(0, "exact", unit_vector(1)),
            chunk(1, "half", unit_vector(1, 2), page=2),
            chunk(2, "orthogonal", unit_vector(3), page=3),
        ],
    )
    repository.insert_document(new_doc("b.pdf", "b"), [chunk(0, "other doc", unit_vector(1, 2, 3, 4))])

    results = repository.search_similar(unit_vector(1), limit=10)

    assert [r.content for r in results] == ["exact", "half", "other doc", "orthogonal"]
    assert results[0].similarity == pytest.approx(1.0, abs=1e-6)
    assert results[1].similarity == pytest.approx(1 / math.sqrt(2), abs=1e-6)
    assert results[-1].similarity == pytest.approx(0.0, abs=1e-6)
    assert results[0].filename == "a.pdf" and results[0].page_number == 1


def test_vector_search_metadata_filters(repository: DocumentRepository) -> None:
    a = repository.insert_document(
        new_doc("a.pdf", "a"),
        [chunk(0, "p1", unit_vector(1), page=1), chunk(1, "p2-3", unit_vector(1, 2), page=2, page_end=3), chunk(2, "p5", unit_vector(1, 3), page=5)],
    )
    repository.insert_document(new_doc("b.pdf", "b"), [chunk(0, "b1", unit_vector(1))])
    q = unit_vector(1)

    assert {r.filename for r in repository.search_similar(q, limit=10, filters=SearchFilters(filenames=("b.pdf",)))} == {"b.pdf"}
    assert {r.content for r in repository.search_similar(q, limit=10, filters=SearchFilters(document_ids=(a,)))} == {"p1", "p2-3", "p5"}
    pages = repository.search_similar(q, limit=10, filters=SearchFilters(filenames=("a.pdf",), page_from=3, page_to=4))
    assert [r.content for r in pages] == ["p2-3"]


def test_resize_embedding_dimension(database_url: str, repository: DocumentRepository) -> None:
    from app.vectorstore.migrations import resize_embedding_dimension

    repository.insert_document(new_doc("a.pdf", "a"), [chunk(0, "x", unit_vector(1))])
    try:
        assert resize_embedding_dimension(database_url, 8) == 1
        assert repository.get_embedding_dimension() == 8
        assert repository.count_documents() == 0
        repository.insert_document(new_doc("b.pdf", "b"), [chunk(0, "small", unit_vector(1, dims=8))])
        assert repository.search_similar(unit_vector(1, dims=8), limit=1)[0].content == "small"
    finally:
        resize_embedding_dimension(database_url, 1536)
    assert repository.get_embedding_dimension() == 1536
    with psycopg.connect(database_url) as conn:
        assert conn.execute("SELECT 1 FROM pg_indexes WHERE indexname = 'document_chunks_embedding_hnsw_idx'").fetchone()


def test_ingestion_lock_prevents_concurrent_runs(database: object, repository: DocumentRepository) -> None:
    other = DocumentRepository(database)  # type: ignore[arg-type]
    with repository.ingestion_lock():
        with pytest.raises(IngestionLockError):
            with other.ingestion_lock():
                pass
    with other.ingestion_lock():  # released afterwards
        pass


# --------------------------------------------------------------------------
# End to end: ingest PDFs into PostgreSQL, then query through the API
# --------------------------------------------------------------------------

ARCH = (
    "The system architecture consists of an ingestion server and a retrieval server. "
    "Both components share one PostgreSQL database with the pgvector extension."
)
SEC = "Access tokens expire after fifteen minutes. Refresh tokens are rotated on every use."


def make_pipeline(directory: Path, repository: DocumentRepository, provider: FakeEmbeddingProvider) -> IngestionPipeline:
    return IngestionPipeline(
        repository=repository,
        embedding_service=EmbeddingService(provider, dimensions=DIMS, batch_size=16, retry_policy=NO_WAIT),
        chunker=TextChunker(WhitespaceTokenizer(), chunk_size=40, chunk_overlap=8),
        pdf_directory=directory,
        embedding_model="fake-embedding",
        max_pdf_size_bytes=10 * 1024 * 1024,
    )


def test_ingestion_end_to_end(tmp_path: Path, database: object, repository: DocumentRepository) -> None:
    make_pdf(tmp_path / "architecture.pdf", ["Introduction page.", ARCH, ""])
    make_pdf(tmp_path / "security.pdf", [SEC])
    provider = FakeEmbeddingProvider(dims=DIMS)
    pipeline = make_pipeline(tmp_path, repository, provider)

    first = pipeline.run()
    assert first.ok and first.inserted == 2
    docs = {d.filename: d for d in repository.list_all_documents()}
    assert docs["architecture.pdf"].total_pages == 3
    with database.connection() as conn:  # type: ignore[attr-defined]
        rows = conn.execute(
            "SELECT d.filename, d.total_chunks, count(c.id) AS n, bool_and(vector_dims(c.embedding) = 1536) AS dims_ok"
            " FROM documents d LEFT JOIN document_chunks c ON c.document_id = d.id GROUP BY d.id"
        ).fetchall()
    for row in rows:
        assert row["total_chunks"] == row["n"] > 0 and row["dims_ok"]

    # Idempotent: no new rows, no embedding calls.
    calls = len(provider.calls)
    second = pipeline.run()
    assert second.unchanged == 2 and second.inserted == second.updated == 0
    assert len(provider.calls) == calls
    assert repository.count_chunks() == sum(r["n"] for r in rows)

    # Modified PDF is reprocessed in place. A removed PDF stays indexed unless pruned.
    make_pdf(tmp_path / "security.pdf", ["Access tokens now expire after five minutes."])
    (tmp_path / "architecture.pdf").unlink()
    kept = pipeline.run()
    assert kept.updated == 1 and kept.deleted == 0 and repository.count_documents() == 2
    third = pipeline.run(prune=True)
    assert third.deleted == 1
    assert [d.filename for d in repository.list_all_documents()] == ["security.pdf"]
    hits = repository.search_similar(bag_of_words_vector("tokens expire minutes", DIMS), limit=5)
    assert hits and all("five minutes" in h.content for h in hits)

    # Query through the HTTP API against the real database.
    make_pdf(tmp_path / "architecture.pdf", ["Introduction page.", ARCH])
    pipeline.run()
    llm = FakeLLM(answer="It has an ingestion server and a retrieval server [architecture.pdf, p. 2].")
    retriever = Retriever(
        embedding_service=EmbeddingService(provider, dimensions=DIMS, retry_policy=NO_WAIT),
        searcher=repository,
        similarity_threshold=0.2,
    )
    service = RAGService(retriever=retriever, context_builder=ContextBuilder(WhitespaceTokenizer(), max_tokens=2000), llm=llm)
    container = ServiceContainer(settings=pipeline_settings(), database=database, documents=repository, rag_service=service)  # type: ignore[arg-type]
    app = create_app(container.settings, container_factory=lambda _: container, configure_logs=False)
    container.close = lambda: None  # type: ignore[method-assign]  # session fixture owns the pool

    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["status"] == "ok" and health["database"]["schema_ready"] is True
        listing = client.get("/api/v1/documents").json()
        assert {d["filename"] for d in listing["documents"]} == {"architecture.pdf", "security.pdf"}
        body = client.post("/api/v1/rag/query", json={"question": "What is the system architecture?", "top_k": 3}).json()

    assert body["success"] is True
    assert body["sources"][0]["document"] == "architecture.pdf"
    # The short intro page and the architecture page share one chunk.
    assert (body["sources"][0]["page"], body["sources"][0]["page_end"]) == (1, 2)
    assert 'document="architecture.pdf" pages="pp. 1-2"' in llm.prompts[0][1]
    assert "ingestion server" in llm.prompts[0][1]


def pipeline_settings():
    from app.config import Settings

    return Settings(_env_file=None, database_url="postgresql://unused/db", anthropic_model="test", anthropic_api_key="test", openai_api_key="test")  # type: ignore[call-arg]
