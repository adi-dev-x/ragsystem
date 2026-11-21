"""Page-by-page text extraction with PyMuPDF."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

import pymupdf

from app.core.exceptions import PDFExtractionError, PDFValidationError
from app.core.logging import get_logger
from app.ingestion.models import ExtractedDocument, PageText

logger = get_logger(__name__)

# Keep ligature/whitespace handling predictable and clip text outside the page box.
_TEXT_FLAGS = pymupdf.TEXT_MEDIABOX_CLIP | pymupdf.TEXT_PRESERVE_WHITESPACE
_METADATA_KEYS = ("title", "author", "subject", "keywords", "creator", "producer", "creationDate", "modDate")

_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(?=[a-z])")
_LINE_BREAK_RE = re.compile(r"[ \t]*\n[ \t]*")
_SPACES_RE = re.compile(r"[ \t  -​  　]+")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_block_text(text: str) -> str:
    """Normalise one text block (roughly a paragraph) into a single clean line."""
    text = unicodedata.normalize("NFKC", text)      # ligatures (ﬁ -> fi), full-width chars
    text = _CONTROL_RE.sub("", text)                  # NUL and friends are rejected by PostgreSQL
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_BREAK_RE.sub(r"\1", text)          # "exam-\nple" -> "example"
    text = _LINE_BREAK_RE.sub(" ", text)              # soft line wraps -> spaces
    text = _SPACES_RE.sub(" ", text)
    return text.strip()


def _page_text(page: pymupdf.Page) -> str:
    # Blocks in natural reading order; block_type 0 is text, 1 is image.
    blocks = page.get_text("blocks", sort=True, flags=_TEXT_FLAGS)
    paragraphs = [clean_block_text(b[4]) for b in blocks if b[6] == 0]
    return "\n\n".join(p for p in paragraphs if p)


def _metadata(doc: pymupdf.Document) -> dict[str, Any]:
    raw = doc.metadata or {}
    return {key: str(raw[key]).strip() for key in _METADATA_KEYS if raw.get(key) and str(raw[key]).strip()}


def extract_pdf(path: Path) -> ExtractedDocument:
    """Extract text from every page. Raises on files that cannot be opened.

    A single unreadable page is logged and treated as empty rather than failing
    the whole document.
    """
    try:
        doc = pymupdf.open(path)
    except Exception as exc:  # PyMuPDF raises several types for corrupt files
        raise PDFValidationError(f"Cannot open PDF {path.name}: {exc}") from exc

    with doc:
        if not doc.is_pdf:
            raise PDFValidationError(f"Not a PDF document: {path.name}")
        if doc.needs_pass:
            raise PDFValidationError(f"PDF is password protected: {path.name}")
        if doc.page_count == 0:
            raise PDFValidationError(f"PDF has no pages: {path.name}")

        pages: list[PageText] = []
        failed_pages: list[int] = []
        for index in range(doc.page_count):
            page_number = index + 1
            try:
                text = _page_text(doc.load_page(index))
            except Exception as exc:
                logger.warning(
                    "page extraction failed; treating page as empty",
                    extra={"file": path.name, "page": page_number, "error": str(exc)},
                )
                text = ""
                failed_pages.append(page_number)
            pages.append(PageText(page_number=page_number, text=text))

        if len(failed_pages) == doc.page_count:
            raise PDFExtractionError(f"Text extraction failed on every page of {path.name}")

        extracted = ExtractedDocument(pages=pages, total_pages=doc.page_count, metadata=_metadata(doc))

    if extracted.empty_pages:
        logger.info(
            "pages without extractable text",
            extra={"file": path.name, "empty_pages": extracted.empty_pages[:50], "count": len(extracted.empty_pages)},
        )
    if not extracted.has_text:
        logger.warning("PDF contains no extractable text", extra={"file": path.name})
    return extracted

