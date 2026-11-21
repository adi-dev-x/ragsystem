"""Records exchanged with the repository layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    id: UUID
    filename: str
    file_hash: str
    file_size: int
    total_pages: int
    total_chunks: int
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class NewDocument:
    """Document fields written on insert or full replacement."""

    filename: str
    file_hash: str
    file_size: int
    total_pages: int
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class NewChunk:
    chunk_index: int
    content: str
    page_number: int
    page_end: int
    token_count: int
    embedding: list[float]


@dataclass(frozen=True, slots=True)
class SearchFilters:
    """Metadata filters applied to vector search."""

    filenames: tuple[str, ...] | None = None
    document_ids: tuple[UUID, ...] | None = None
    page_from: int | None = None
    page_to: int | None = None

    def is_empty(self) -> bool:
        return not (self.filenames or self.document_ids or self.page_from or self.page_to)


@dataclass(frozen=True, slots=True)
class ChunkSearchResult:
    chunk_id: UUID
    document_id: UUID
    filename: str
    chunk_index: int
    page_number: int
    page_end: int
    content: str
    token_count: int
    similarity: float  # cosine similarity, 1 - cosine distance


@dataclass(frozen=True, slots=True)
class DatabaseHealth:
    ok: bool
    latency_ms: float | None = None
    pgvector_version: str | None = None
    schema_ready: bool = False
    error: str | None = None
