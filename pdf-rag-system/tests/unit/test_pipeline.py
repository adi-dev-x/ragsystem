"""Ingestion pipeline behaviour with an in-memory repository and fake embeddings."""

from pathlib import Path

import pytest

from app.embeddings.embedding_service import EmbeddingService
from app.ingestion.chunker import TextChunker
from app.ingestion.pipeline import IngestionPipeline, dedupe_chunks
from app.ingestion.models import TextChunk
from tests.conftest import NO_WAIT
from tests.helpers import FakeEmbeddingProvider, InMemoryRepository, WhitespaceTokenizer, make_pdf

ARCH = "The ingestion server extracts text from PDF files. Chunks are embedded and stored in PostgreSQL."
API = "The retrieval server exposes a FastAPI endpoint. It searches pgvector with cosine similarity."


@pytest.fixture
def repo() -> InMemoryRepository:
    return InMemoryRepository()


@pytest.fixture
def provider() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider(dims=64)


def make_pipeline(directory: Path, repo: InMemoryRepository, provider: FakeEmbeddingProvider, *, model: str = "fake-embedding", chunk_size: int = 40) -> IngestionPipeline:
    return IngestionPipeline(
        repository=repo,
        embedding_service=EmbeddingService(provider, dimensions=64, batch_size=4, retry_policy=NO_WAIT),
        chunker=TextChunker(WhitespaceTokenizer(), chunk_size=chunk_size, chunk_overlap=8),
        pdf_directory=directory,
        embedding_model=model,
        max_pdf_size_bytes=10 * 1024 * 1024,
    )


def test_first_run_ingests_all_documents(tmp_path: Path, repo: InMemoryRepository, provider: FakeEmbeddingProvider) -> None:
    make_pdf(tmp_path / "arch.pdf", [ARCH, API])
    make_pdf(tmp_path / "api.pdf", [API])

    report = make_pipeline(tmp_path, repo, provider).run()

    assert report.ok and report.inserted == 2 and report.discovered == 2
    arch = repo.by_filename("arch.pdf")
    assert arch is not None and arch.total_pages == 2 and arch.total_chunks >= 1
    assert report.chunks_written == sum(d.total_chunks for d in repo.documents.values())
    assert provider.texts_embedded == report.chunks_written


def test_second_run_is_idempotent(tmp_path: Path, repo: InMemoryRepository, provider: FakeEmbeddingProvider) -> None:
    make_pdf(tmp_path / "arch.pdf", [ARCH])
    make_pdf(tmp_path / "api.pdf", [API])
    pipeline = make_pipeline(tmp_path, repo, provider)
    pipeline.run()
    snapshot = {d.id: (d.file_hash, d.total_chunks) for d in repo.documents.values()}
    calls_after_first = len(provider.calls)

    report = pipeline.run()

    assert report.unchanged == 2 and report.inserted == report.updated == report.deleted == 0
    assert len(provider.calls) == calls_after_first, "no embedding calls for unchanged PDFs"
    assert {d.id: (d.file_hash, d.total_chunks) for d in repo.documents.values()} == snapshot
    assert repo.count_documents() == 2


def test_modified_pdf_is_reprocessed(tmp_path: Path, repo: InMemoryRepository, provider: FakeEmbeddingProvider) -> None:
    make_pdf(tmp_path / "arch.pdf", [ARCH])
    make_pdf(tmp_path / "api.pdf", [API])
    pipeline = make_pipeline(tmp_path, repo, provider)
    pipeline.run()
    original = repo.by_filename("arch.pdf")
    provider.calls.clear()

    make_pdf(tmp_path / "arch.pdf", ["The architecture now uses a message queue between services."])
    report = pipeline.run()

    updated = repo.by_filename("arch.pdf")
    assert report.updated == 1 and report.unchanged == 1
    assert updated.id == original.id and updated.file_hash != original.file_hash
    assert repo.chunk_texts("arch.pdf") == ["The architecture now uses a message queue between services."]
    assert all("message queue" in t for batch in provider.calls for t in batch), "only the changed PDF is embedded"


def test_removed_pdf_stays_indexed_by_default(tmp_path: Path, repo: InMemoryRepository, provider: FakeEmbeddingProvider) -> None:
    make_pdf(tmp_path / "arch.pdf", [ARCH])
    make_pdf(tmp_path / "api.pdf", [API])
    pipeline = make_pipeline(tmp_path, repo, provider)
    pipeline.run()
    chunks_before = repo.chunk_texts("api.pdf")
    original_bytes = (tmp_path / "api.pdf").read_bytes()
    provider.calls.clear()

    (tmp_path / "api.pdf").unlink()
    report = pipeline.run()

    assert report.ok and report.deleted == 0
    assert repo.chunk_texts("api.pdf") == chunks_before
    assert report.planned["kept_missing"] == 1
    assert provider.calls == []

    # Putting the same file back is recognised by its hash: still no embedding calls.
    (tmp_path / "api.pdf").write_bytes(original_bytes)
    assert pipeline.run().unchanged == 2 and provider.calls == []


def test_emptied_folder_does_not_wipe_the_index(tmp_path: Path, repo: InMemoryRepository, provider: FakeEmbeddingProvider) -> None:
    make_pdf(tmp_path / "arch.pdf", [ARCH])
    make_pdf(tmp_path / "api.pdf", [API])
    pipeline = make_pipeline(tmp_path, repo, provider)
    pipeline.run()

    (tmp_path / "arch.pdf").unlink()
    (tmp_path / "api.pdf").unlink()
    report = pipeline.run()

    assert report.deleted == 0 and repo.count_documents() == 2


def test_prune_removes_documents_whose_pdf_is_gone(tmp_path: Path, repo: InMemoryRepository, provider: FakeEmbeddingProvider) -> None:
    make_pdf(tmp_path / "arch.pdf", [ARCH])
    make_pdf(tmp_path / "api.pdf", [API])
    pipeline = make_pipeline(tmp_path, repo, provider)
    pipeline.run()

    (tmp_path / "api.pdf").unlink()
    report = pipeline.run(prune=True)

    assert report.deleted == 1
    assert repo.by_filename("api.pdf") is None
    assert [d.filename for d in repo.list_all_documents()] == ["arch.pdf"]


def test_renamed_pdf_is_not_re_embedded(tmp_path: Path, repo: InMemoryRepository, provider: FakeEmbeddingProvider) -> None:
    make_pdf(tmp_path / "arch.pdf", [ARCH])
    pipeline = make_pipeline(tmp_path, repo, provider)
    pipeline.run()
    provider.calls.clear()

    (tmp_path / "arch.pdf").rename(tmp_path / "architecture.pdf")
    report = pipeline.run()

    assert report.renamed == 1 and provider.calls == []
    assert repo.by_filename("architecture.pdf") is not None and repo.by_filename("arch.pdf") is None


def test_duplicate_content_is_indexed_once(tmp_path: Path, repo: InMemoryRepository, provider: FakeEmbeddingProvider) -> None:
    make_pdf(tmp_path / "a.pdf", [ARCH])
    (tmp_path / "copy.pdf").write_bytes((tmp_path / "a.pdf").read_bytes())

    report = make_pipeline(tmp_path, repo, provider).run()

    assert report.inserted == 1 and report.duplicates == 1
    assert repo.count_documents() == 1


def test_empty_pdf_is_recorded_without_embedding_calls(tmp_path: Path, repo: InMemoryRepository, provider: FakeEmbeddingProvider) -> None:
    make_pdf(tmp_path / "blank.pdf", ["", ""])

    report = make_pipeline(tmp_path, repo, provider).run()

    blank = repo.by_filename("blank.pdf")
    assert report.ok and blank is not None
    assert blank.total_pages == 2 and blank.total_chunks == 0
    assert provider.calls == []
    assert blank.metadata["empty_pages"] == [1, 2]

    # Still idempotent on the next run.
    assert make_pipeline(tmp_path, repo, provider).run().unchanged == 1


def test_invalid_pdf_fails_without_blocking_others(tmp_path: Path, repo: InMemoryRepository, provider: FakeEmbeddingProvider) -> None:
    make_pdf(tmp_path / "good.pdf", [ARCH])
    (tmp_path / "bad.pdf").write_bytes(b"%PDF-1.4 garbage")
    (tmp_path / "text.pdf").write_text("not a pdf")

    report = make_pipeline(tmp_path, repo, provider).run()

    assert report.inserted == 1 and report.failed == 2
    assert set(report.errors) == {"bad.pdf", "text.pdf"}
    assert not report.ok


def test_embedding_failure_keeps_previous_version(tmp_path: Path, repo: InMemoryRepository, provider: FakeEmbeddingProvider) -> None:
    make_pdf(tmp_path / "arch.pdf", [ARCH])
    pipeline = make_pipeline(tmp_path, repo, provider)
    pipeline.run()
    before = repo.chunk_texts("arch.pdf")

    make_pdf(tmp_path / "arch.pdf", ["POISON content that the provider rejects."])
    provider.fail_on = "POISON"
    report = pipeline.run()

    assert report.failed == 1 and "arch.pdf" in report.errors
    assert repo.chunk_texts("arch.pdf") == before


def test_embedding_model_change_reprocesses(tmp_path: Path, repo: InMemoryRepository, provider: FakeEmbeddingProvider) -> None:
    make_pdf(tmp_path / "arch.pdf", [ARCH])
    make_pipeline(tmp_path, repo, provider, model="model-a").run()

    report = make_pipeline(tmp_path, repo, provider, model="model-b").run()

    assert report.updated == 1
    assert repo.by_filename("arch.pdf").embedding_model == "model-b"


def test_dry_run_writes_nothing(tmp_path: Path, repo: InMemoryRepository, provider: FakeEmbeddingProvider) -> None:
    make_pdf(tmp_path / "arch.pdf", [ARCH])
    report = make_pipeline(tmp_path, repo, provider).run(dry_run=True)
    assert report.dry_run and repo.count_documents() == 0 and provider.calls == []


def test_retryable_embedding_errors_are_retried(tmp_path: Path, repo: InMemoryRepository) -> None:
    flaky = FakeEmbeddingProvider(dims=64, fail_times=2)
    make_pdf(tmp_path / "arch.pdf", [ARCH])
    report = make_pipeline(tmp_path, repo, flaky).run()
    assert report.ok and report.inserted == 1
    assert len(flaky.calls) == 3


def test_dedupe_chunks_drops_repeated_text() -> None:
    chunks = [
        TextChunk(0, "Confidential footer", 1, 1, 2),
        TextChunk(1, "Real content", 2, 2, 2),
        TextChunk(2, "confidential   FOOTER", 3, 3, 2),
    ]
    assert [(c.chunk_index, c.content) for c in dedupe_chunks(chunks)] == [(0, "Confidential footer"), (1, "Real content")]
