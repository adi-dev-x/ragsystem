"""Ingestion pipeline: sync the PDF directory into PostgreSQL.

    discover -> plan -> delete -> rename -> (extract -> chunk -> embed -> store)*

Embeddings are generated *before* any write, and each document's rows are
written in one transaction, so a failure (API error, crash) leaves the
previously indexed version untouched.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol, Sequence
from uuid import UUID

from app.core.exceptions import ConfigurationError, RAGError
from app.core.logging import get_logger
from app.embeddings.embedding_service import EmbeddingService
from app.ingestion.chunker import TextChunker
from app.ingestion.extractor import extract_pdf
from app.ingestion.loader import discover_pdfs
from app.ingestion.models import ExtractedDocument, PDFFile, TextChunk
from app.ingestion.sync import IngestionConfig, ProcessTask, SyncPlan, build_sync_plan
from app.vectorstore.models import DocumentRecord, NewChunk, NewDocument

logger = get_logger(__name__)


class IngestionRepository(Protocol):
    """The subset of DocumentRepository the pipeline needs."""

    def list_all_documents(self) -> list[DocumentRecord]: ...
    def insert_document(self, document: NewDocument, chunks: Sequence[NewChunk]) -> UUID: ...
    def replace_document(self, document_id: UUID, document: NewDocument, chunks: Sequence[NewChunk]) -> None: ...
    def rename_documents(self, renames: Sequence[tuple[UUID, str]]) -> None: ...
    def delete_documents(self, document_ids: Sequence[UUID]) -> int: ...
    def ingestion_lock(self) -> AbstractContextManager[None]: ...


@dataclass(slots=True)
class IngestionReport:
    discovered: int = 0
    unchanged: int = 0
    inserted: int = 0
    updated: int = 0
    renamed: int = 0
    deleted: int = 0
    duplicates: int = 0
    failed: int = 0
    chunks_written: int = 0
    embedding_requests: int = 0
    embedding_tokens: int = 0
    duration_seconds: float = 0.0
    dry_run: bool = False
    planned: dict[str, int] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.failed == 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def dedupe_chunks(chunks: Sequence[TextChunk]) -> list[TextChunk]:
    """Drop chunks whose normalised text repeats an earlier chunk (e.g. boilerplate pages)."""
    seen: set[str] = set()
    unique: list[TextChunk] = []
    for chunk in chunks:
        key = " ".join(chunk.content.split()).casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(
            TextChunk(
                chunk_index=len(unique),
                content=chunk.content,
                page_number=chunk.page_number,
                page_end=chunk.page_end,
                token_count=chunk.token_count,
            )
        )
    return unique


class IngestionPipeline:
    def __init__(
        self,
        *,
        repository: IngestionRepository,
        embedding_service: EmbeddingService | None,
        chunker: TextChunker,
        pdf_directory: Path,
        embedding_model: str,
        max_pdf_size_bytes: int,
        extractor: Callable[[Path], ExtractedDocument] = extract_pdf,
    ) -> None:
        self._repo = repository
        self._embedder = embedding_service
        self._chunker = chunker
        self._directory = pdf_directory
        self._max_size = max_pdf_size_bytes
        self._extract = extractor
        self._config = IngestionConfig(
            embedding_model=embedding_model,
            chunk_size=chunker.chunk_size,
            chunk_overlap=chunker.chunk_overlap,
        )

    def run(self, *, force: bool = False, dry_run: bool = False, prune: bool = False) -> IngestionReport:
        """Sync the PDF directory into the database.

        Documents whose PDF is no longer on disk stay indexed unless ``prune`` is set.
        """
        started = time.perf_counter()
        report = IngestionReport(dry_run=dry_run)
        logger.info(
            "ingestion started",
            extra={"directory": str(self._directory), "force": force, "dry_run": dry_run, "prune": prune},
        )

        discovery = discover_pdfs(self._directory, max_size_bytes=self._max_size)
        report.discovered = len(discovery.files)
        for filename, reason in discovery.invalid.items():
            report.failed += 1
            report.errors[filename] = reason

        with self._repo.ingestion_lock():
            plan = build_sync_plan(
                discovery.files,
                self._repo.list_all_documents(),
                self._config,
                invalid_filenames=discovery.invalid.keys(),
                force=force,
                prune=prune,
            )
            self._log_plan(plan)
            report.unchanged = len(plan.unchanged)
            report.duplicates = len(plan.duplicates)
            report.planned = plan.summary()

            if dry_run:
                report.duration_seconds = round(time.perf_counter() - started, 3)
                return report
            if plan.to_process and self._embedder is None:
                raise ConfigurationError("An embedding service is required to ingest new or modified PDFs")

            self._apply_deletions(plan, report)
            self._apply_renames(plan, report)
            for task in plan.to_process:
                self._process(task, report)

        report.duration_seconds = round(time.perf_counter() - started, 3)
        log = logger.info if report.ok else logger.warning
        log("ingestion finished", extra={k: v for k, v in report.to_dict().items() if k != "errors"})
        return report

    # ------------------------------------------------------------------

    def _log_plan(self, plan: SyncPlan) -> None:
        logger.info("ingestion plan", extra=plan.summary())
        for path, original in plan.duplicates.items():
            logger.warning("duplicate PDF content skipped", extra={"document": path, "same_as": original})
        for name in plan.kept_invalid:
            logger.warning("keeping existing index for unreadable file", extra={"document": name})
        for name in plan.kept_missing:
            logger.info(
                "file no longer on disk; keeping its index (use --prune to remove)", extra={"document": name}
            )
        for f in plan.unchanged:
            logger.debug("unchanged, skipping", extra={"document": f.filename})

    def _apply_deletions(self, plan: SyncPlan, report: IngestionReport) -> None:
        if not plan.deletions:
            return
        report.deleted = self._repo.delete_documents([d.id for d in plan.deletions])
        for doc in plan.deletions:
            logger.info(
                "removed document no longer on disk",
                extra={"document": doc.filename, "document_id": str(doc.id), "chunks": doc.total_chunks},
            )

    def _apply_renames(self, plan: SyncPlan, report: IngestionReport) -> None:
        if not plan.renames:
            return
        self._repo.rename_documents([(doc_id, new) for doc_id, _, new in plan.renames])
        report.renamed = len(plan.renames)
        for doc_id, old, new in plan.renames:
            logger.info("document renamed", extra={"document_id": str(doc_id), "from": old, "to": new})

    def _process(self, task: ProcessTask, report: IngestionReport) -> None:
        file = task.file
        started = time.perf_counter()
        try:
            document, chunks, embed_stats = self._build(file)
            if task.existing_document_id is None:
                document_id = self._repo.insert_document(document, chunks)
                report.inserted += 1
            else:
                document_id = task.existing_document_id
                self._repo.replace_document(document_id, document, chunks)
                report.updated += 1
        except RAGError as exc:
            report.failed += 1
            report.errors[file.filename] = exc.message
            logger.error("failed to ingest document", extra={"document": file.filename, "error": exc.message, "code": exc.code})
            return
        except Exception as exc:  # keep going with the remaining files
            report.failed += 1
            report.errors[file.filename] = f"{exc.__class__.__name__}: {exc}"
            logger.exception("unexpected error ingesting document", extra={"document": file.filename})
            return

        report.chunks_written += len(chunks)
        report.embedding_requests += embed_stats[0]
        report.embedding_tokens += embed_stats[1]
        logger.info(
            "document ingested",
            extra={
                "document": file.filename,
                "document_id": str(document_id),
                "reason": task.reason,
                "pages": document.total_pages,
                "chunks": len(chunks),
                "embedding_tokens": embed_stats[1],
                "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            },
        )

    def _build(self, file: PDFFile) -> tuple[NewDocument, list[NewChunk], tuple[int, int]]:
        extracted = self._extract(file.path)
        text_chunks = dedupe_chunks(self._chunker.split_pages(extracted.pages))
        text_chunks = [c for c in text_chunks if c.content.strip()]

        requests = tokens = 0
        vectors: list[list[float]] = []
        if text_chunks:
            assert self._embedder is not None
            result = self._embedder.embed_texts([c.content for c in text_chunks])
            vectors, requests, tokens = result.vectors, result.requests, result.total_tokens
        else:
            logger.warning("no text chunks produced; storing document without chunks", extra={"document": file.filename})

        document = NewDocument(
            filename=file.filename,
            file_hash=file.sha256,
            file_size=file.size,
            total_pages=extracted.total_pages,
            embedding_model=self._config.embedding_model,
            chunk_size=self._config.chunk_size,
            chunk_overlap=self._config.chunk_overlap,
            metadata={**extracted.metadata, "empty_pages": extracted.empty_pages},
        )
        chunks = [
            NewChunk(
                chunk_index=c.chunk_index,
                content=c.content,
                page_number=c.page_number,
                page_end=c.page_end,
                token_count=c.token_count,
                embedding=vector,
            )
            for c, vector in zip(text_chunks, vectors, strict=True)
        ]
        return document, chunks, (requests, tokens)
