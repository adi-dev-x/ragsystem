"""Test doubles and builders shared by unit and integration tests."""

from __future__ import annotations

import hashlib
import math
import re
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

import pymupdf

from app.core.exceptions import EmbeddingError
from app.embeddings.base import EmbeddingBatch, EmbeddingProvider
from app.llm.base import LLMProvider, LLMResponse
from app.vectorstore.models import (
    ChunkSearchResult,
    DatabaseHealth,
    DocumentRecord,
    NewChunk,
    NewDocument,
    SearchFilters,
)

# --------------------------------------------------------------------------
# Tokenizer
# --------------------------------------------------------------------------


class WhitespaceTokenizer:
    """One token per whitespace-separated word; makes chunk maths exact in tests."""

    def __init__(self) -> None:
        self._ids: dict[str, int] = {}
        self._words: list[str] = []

    def encode(self, text: str) -> list[int]:
        out = []
        for word in text.split():
            if word not in self._ids:
                self._ids[word] = len(self._words)
                self._words.append(word)
            out.append(self._ids[word])
        return out

    def decode(self, tokens: Sequence[int]) -> str:
        return " ".join(self._words[t] for t in tokens)

    def count(self, text: str) -> int:
        return len(text.split())


# --------------------------------------------------------------------------
# PDFs
# --------------------------------------------------------------------------


def make_pdf(path: Path, pages: Sequence[str], *, title: str | None = None) -> Path:
    """Write a text-only PDF; an empty string produces a blank page."""
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page(width=595, height=842)
        if text:
            overflow = page.insert_textbox(pymupdf.Rect(40, 40, 555, 802), text, fontsize=8)
            assert overflow >= 0, "test text does not fit on one page"
    if title:
        doc.set_metadata({"title": title})
    doc.save(path)
    doc.close()
    return path


# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9]+")


def bag_of_words_vector(text: str, dims: int) -> list[float]:
    """Deterministic embedding: hashed word counts, L2-normalised.

    Texts sharing words get high cosine similarity, which is enough to test
    ranking without calling a real model.
    """
    vector = [0.0] * dims
    for word in _WORD_RE.findall(text.lower()):
        index = int(hashlib.md5(word.encode()).hexdigest(), 16) % dims
        vector[index] += 1.0
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        vector[0] = 1.0
        return vector
    return [v / norm for v in vector]


class FakeEmbeddingProvider(EmbeddingProvider):
    name = "fake"

    def __init__(self, dims: int = 64, *, model: str = "fake-embedding", fail_times: int = 0, fail_on: str | None = None) -> None:
        self.model = model
        self.dims = dims
        self.calls: list[list[str]] = []
        self._fail_times = fail_times
        self.fail_on = fail_on  # raise a permanent error if any input contains this text

    @property
    def texts_embedded(self) -> int:
        return sum(len(c) for c in self.calls)

    def embed_batch(self, texts: Sequence[str]) -> EmbeddingBatch:
        self.calls.append(list(texts))
        if self._fail_times > 0:
            self._fail_times -= 1
            raise EmbeddingError("simulated rate limit", retryable=True, retry_after=0)
        if self.fail_on and any(self.fail_on in t for t in texts):
            raise EmbeddingError("simulated permanent failure", retryable=False)
        return EmbeddingBatch(
            vectors=[bag_of_words_vector(t, self.dims) for t in texts],
            total_tokens=sum(len(t.split()) for t in texts),
        )


# --------------------------------------------------------------------------
# LLM
# --------------------------------------------------------------------------


class FakeLLM(LLMProvider):
    def __init__(self, *, name: str = "claude", model: str = "fake-model", answer: str = "Answer [doc.pdf, p. 1]", error: Exception | None = None) -> None:
        super().__init__()
        self.name = name
        self.model = model
        self.answer = answer
        self.error = error
        self.prompts: list[tuple[str, str]] = []

    def _generate_once(self, *, system_prompt: str, user_prompt: str) -> LLMResponse:
        self.prompts.append((system_prompt, user_prompt))
        if self.error:
            raise self.error
        return LLMResponse(
            text=self.answer, provider=self.name, model=self.model, input_tokens=1200, output_tokens=250, stop_reason="end_turn"
        )


# --------------------------------------------------------------------------
# Repository
# --------------------------------------------------------------------------


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class InMemoryRepository:
    """Implements the repository methods used by the pipeline, retriever and API."""

    def __init__(self) -> None:
        self.documents: dict[UUID, DocumentRecord] = {}
        self.chunks: dict[UUID, list[tuple[UUID, NewChunk]]] = {}
        self.locked = False

    # reads
    def list_all_documents(self) -> list[DocumentRecord]:
        return sorted(self.documents.values(), key=lambda d: d.filename)

    def list_documents(self, *, limit: int = 100, offset: int = 0) -> list[DocumentRecord]:
        return self.list_all_documents()[offset: offset + limit]

    def count_documents(self) -> int:
        return len(self.documents)

    def by_filename(self, filename: str) -> DocumentRecord | None:
        return next((d for d in self.documents.values() if d.filename == filename), None)

    def chunk_texts(self, filename: str) -> list[str]:
        doc = self.by_filename(filename)
        return [c.content for _, c in self.chunks.get(doc.id, [])] if doc else []

    # writes
    def _record(self, doc_id: UUID, document: NewDocument, n_chunks: int, created: datetime | None = None) -> DocumentRecord:
        now = datetime.now(timezone.utc)
        return DocumentRecord(
            id=doc_id,
            filename=document.filename,
            file_hash=document.file_hash,
            file_size=document.file_size,
            total_pages=document.total_pages,
            total_chunks=n_chunks,
            embedding_model=document.embedding_model,
            chunk_size=document.chunk_size,
            chunk_overlap=document.chunk_overlap,
            metadata=dict(document.metadata),
            created_at=created or now,
            updated_at=now,
        )

    def _check_unique(self, document: NewDocument, exclude: UUID | None = None) -> None:
        for d in self.documents.values():
            if d.id == exclude:
                continue
            assert d.filename != document.filename, "UNIQUE(filename) violated"
            assert d.file_hash != document.file_hash, "UNIQUE(file_hash) violated"

    def insert_document(self, document: NewDocument, chunks: Sequence[NewChunk]) -> UUID:
        self._check_unique(document)
        doc_id = uuid.uuid4()
        self.documents[doc_id] = self._record(doc_id, document, len(chunks))
        self.chunks[doc_id] = [(uuid.uuid4(), c) for c in chunks]
        return doc_id

    def replace_document(self, document_id: UUID, document: NewDocument, chunks: Sequence[NewChunk]) -> None:
        self._check_unique(document, exclude=document_id)
        created = self.documents[document_id].created_at
        self.documents[document_id] = self._record(document_id, document, len(chunks), created)
        self.chunks[document_id] = [(uuid.uuid4(), c) for c in chunks]

    def rename_documents(self, renames: Sequence[tuple[UUID, str]]) -> None:
        for doc_id, name in renames:
            d = self.documents[doc_id]
            self.documents[doc_id] = DocumentRecord(**{**{f: getattr(d, f) for f in d.__slots__}, "filename": name})
        names = [d.filename for d in self.documents.values()]
        assert len(names) == len(set(names)), "UNIQUE(filename) violated after rename"

    def delete_documents(self, document_ids: Sequence[UUID]) -> int:
        n = 0
        for doc_id in document_ids:
            if self.documents.pop(doc_id, None):
                self.chunks.pop(doc_id, None)
                n += 1
        return n

    @contextmanager
    def ingestion_lock(self) -> Iterator[None]:
        assert not self.locked
        self.locked = True
        try:
            yield
        finally:
            self.locked = False

    # search
    def search_similar(
        self, embedding: Sequence[float], *, limit: int, filters: SearchFilters | None = None, ef_search: int = 100
    ) -> list[ChunkSearchResult]:
        results = []
        for doc_id, chunks in self.chunks.items():
            doc = self.documents[doc_id]
            if filters:
                if filters.filenames and doc.filename not in filters.filenames:
                    continue
                if filters.document_ids and doc_id not in filters.document_ids:
                    continue
            for chunk_id, c in chunks:
                if filters and filters.page_from and c.page_end < filters.page_from:
                    continue
                if filters and filters.page_to and c.page_number > filters.page_to:
                    continue
                results.append(
                    ChunkSearchResult(
                        chunk_id=chunk_id,
                        document_id=doc_id,
                        filename=doc.filename,
                        chunk_index=c.chunk_index,
                        page_number=c.page_number,
                        page_end=c.page_end,
                        content=c.content,
                        token_count=c.token_count,
                        similarity=_cosine(embedding, c.embedding),
                    )
                )
        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:limit]


class FakeHealth:
    def __init__(self, health: DatabaseHealth | None = None) -> None:
        self.health = health or DatabaseHealth(ok=True, latency_ms=1.2, pgvector_version="0.8.0", schema_ready=True)
        self.closed = False

    def check_health(self) -> DatabaseHealth:
        return self.health

    def close(self) -> None:
        self.closed = True


def search_result(content: str, similarity: float, *, filename: str = "doc.pdf", page: int = 1, page_end: int | None = None, **kw: Any) -> ChunkSearchResult:
    return ChunkSearchResult(
        chunk_id=kw.get("chunk_id", uuid.uuid4()),
        document_id=kw.get("document_id", uuid.uuid4()),
        filename=filename,
        chunk_index=kw.get("chunk_index", 0),
        page_number=page,
        page_end=page_end or page,
        content=content,
        token_count=len(content.split()),
        similarity=similarity,
    )
