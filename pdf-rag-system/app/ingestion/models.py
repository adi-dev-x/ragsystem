"""Data passed between ingestion stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class PDFFile:
    """A PDF discovered on disk."""

    path: Path          # absolute path
    filename: str       # POSIX path relative to the PDF directory (stored in the DB)
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class PageText:
    page_number: int    # 1-based
    text: str

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    pages: list[PageText]
    total_pages: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def empty_pages(self) -> list[int]:
        return [p.page_number for p in self.pages if p.is_empty]

    @property
    def has_text(self) -> bool:
        return any(not p.is_empty for p in self.pages)


@dataclass(frozen=True, slots=True)
class TextChunk:
    chunk_index: int
    content: str
    page_number: int    # first page covered
    page_end: int       # last page covered
    token_count: int
