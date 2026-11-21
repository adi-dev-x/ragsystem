"""Discover PDF files, run cheap validation, and hash their contents."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from app.core.exceptions import IngestionError, PDFValidationError
from app.core.logging import get_logger
from app.ingestion.models import PDFFile

logger = get_logger(__name__)

PDF_MAGIC = b"%PDF-"
_HASH_CHUNK_BYTES = 1024 * 1024


def compute_sha256(path: Path) -> str:
    """SHA-256 hex digest of a file, streamed so large PDFs aren't loaded into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_pdf_file(path: Path, *, max_size_bytes: int) -> int:
    """Check the file is a non-empty PDF within the size limit. Returns its size."""
    if not path.is_file():
        raise PDFValidationError(f"Not a regular file: {path.name}")
    size = path.stat().st_size
    if size == 0:
        raise PDFValidationError(f"File is empty: {path.name}")
    if size > max_size_bytes:
        raise PDFValidationError(
            f"File exceeds size limit ({size} bytes > {max_size_bytes} bytes): {path.name}"
        )
    with path.open("rb") as handle:
        # The PDF header may be preceded by a little junk; the spec allows 1024 bytes.
        head = handle.read(1024)
    if PDF_MAGIC not in head:
        raise PDFValidationError(f"Missing %PDF- header, not a PDF: {path.name}")
    return size


@dataclass(slots=True)
class DiscoveryResult:
    files: list[PDFFile] = field(default_factory=list)
    invalid: dict[str, str] = field(default_factory=dict)  # filename -> reason


def discover_pdfs(directory: Path, *, max_size_bytes: int) -> DiscoveryResult:
    """Find ``*.pdf`` files (recursively, case-insensitive), validate and hash them."""
    if not directory.exists():
        raise IngestionError(f"PDF directory does not exist: {directory}")
    if not directory.is_dir():
        raise IngestionError(f"PDF directory is not a directory: {directory}")

    result = DiscoveryResult()
    candidates = sorted(
        p for p in directory.rglob("*")
        if p.suffix.lower() == ".pdf" and not any(part.startswith(".") for part in p.relative_to(directory).parts)
    )
    for path in candidates:
        filename = path.relative_to(directory).as_posix()
        try:
            size = validate_pdf_file(path, max_size_bytes=max_size_bytes)
            sha256 = compute_sha256(path)
        except PDFValidationError as exc:
            logger.error("invalid PDF skipped", extra={"document": filename, "reason": exc.message})
            result.invalid[filename] = exc.message
            continue
        except OSError as exc:
            logger.error("could not read PDF", extra={"document": filename, "reason": str(exc)})
            result.invalid[filename] = f"Could not read file: {exc.strerror or exc}"
            continue
        result.files.append(PDFFile(path=path.resolve(), filename=filename, size=size, sha256=sha256))

    logger.info(
        "PDF discovery complete",
        extra={"directory": str(directory), "valid": len(result.files), "invalid": len(result.invalid)},
    )
    return result
