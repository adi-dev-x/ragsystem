"""Request and response schemas for the HTTP API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# --------------------------------------------------------------------------
# Query
# --------------------------------------------------------------------------


class QueryFilters(BaseModel):
    """Metadata filters; all supplied conditions must match."""

    model_config = ConfigDict(extra="forbid")

    filenames: list[str] | None = Field(
        default=None, max_length=100, description="Restrict to these document filenames."
    )
    document_ids: list[UUID] | None = Field(default=None, max_length=100)
    page_from: int | None = Field(default=None, ge=1, description="Chunks ending on or after this page.")
    page_to: int | None = Field(default=None, ge=1, description="Chunks starting on or before this page.")

    @field_validator("filenames")
    @classmethod
    def _clean_filenames(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        cleaned = [v.strip() for v in value if v and v.strip()]
        return cleaned or None

    @model_validator(mode="after")
    def _check_pages(self) -> "QueryFilters":
        if self.page_from and self.page_to and self.page_from > self.page_to:
            raise ValueError("page_from must be less than or equal to page_to")
        return self


class QueryRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        json_schema_extra={"examples": [{"question": "Explain the system architecture", "top_k": 5}]},
    )

    question: str = Field(min_length=1, max_length=4000, description="Natural-language question.")
    top_k: int | None = Field(default=None, ge=1, le=50, description="Number of chunks to retrieve (default TOP_K).")
    similarity_threshold: float | None = Field(
        default=None, ge=0, le=1, description="Minimum cosine similarity (default SIMILARITY_THRESHOLD)."
    )
    filters: QueryFilters | None = None


class SourceReference(BaseModel):
    id: int = Field(description="Citation number used in the answer, e.g. [1]")
    document: str
    document_id: UUID
    page: int
    page_end: int
    chunk_id: UUID
    chunk_index: int
    relevance_score: float
    content: str


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class QueryResponse(BaseModel):
    success: Literal[True] = True
    question: str
    answer: str
    sources: list[SourceReference]
    model: str
    provider: str
    usage: Usage


# --------------------------------------------------------------------------
# Documents
# --------------------------------------------------------------------------


class DocumentInfo(BaseModel):
    id: UUID
    filename: str
    file_hash: str
    file_size: int
    total_pages: int
    total_chunks: int
    embedding_model: str
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class DocumentListResponse(BaseModel):
    success: Literal[True] = True
    total: int
    limit: int
    offset: int
    documents: list[DocumentInfo]


# --------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------


class DatabaseStatus(BaseModel):
    status: Literal["ok", "error"]
    latency_ms: float | None = None
    pgvector_version: str | None = None
    schema_ready: bool = False
    error: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    api: Literal["ok"] = "ok"
    database: DatabaseStatus
    llm_provider: str
    embedding_model: str


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class ErrorBody(BaseModel):
    code: str
    message: str
    details: Any | None = None


class ErrorResponse(BaseModel):
    success: Literal[False] = False
    error: ErrorBody
    request_id: str | None = None
