import hashlib
from pathlib import Path

import pytest

from app.core.exceptions import IngestionError, PDFValidationError
from app.ingestion.loader import compute_sha256, discover_pdfs, validate_pdf_file
from tests.helpers import make_pdf

MAX = 10 * 1024 * 1024


def test_sha256_matches_hashlib_and_detects_changes(tmp_path: Path) -> None:
    pdf = make_pdf(tmp_path / "a.pdf", ["version one"])
    first = compute_sha256(pdf)
    assert first == hashlib.sha256(pdf.read_bytes()).hexdigest()

    make_pdf(pdf, ["version two"])
    assert compute_sha256(pdf) != first


def test_identical_content_has_identical_hash(tmp_path: Path) -> None:
    a = make_pdf(tmp_path / "a.pdf", ["same"])
    b = tmp_path / "b.pdf"
    b.write_bytes(a.read_bytes())
    assert compute_sha256(a) == compute_sha256(b)


def test_discovery_finds_nested_pdfs_and_reports_invalid(tmp_path: Path) -> None:
    make_pdf(tmp_path / "top.pdf", ["top"])
    make_pdf(tmp_path / "sub" / "Nested.PDF", ["nested"])
    (tmp_path / "notes.txt").write_text("ignore me")
    (tmp_path / "fake.pdf").write_text("plain text pretending to be a pdf")
    (tmp_path / "empty.pdf").write_bytes(b"")
    make_pdf(tmp_path / ".hidden" / "skip.pdf", ["hidden"])

    result = discover_pdfs(tmp_path, max_size_bytes=MAX)

    assert sorted(f.filename for f in result.files) == ["sub/Nested.PDF", "top.pdf"]
    assert set(result.invalid) == {"fake.pdf", "empty.pdf"}
    assert all(len(f.sha256) == 64 for f in result.files)


def test_missing_directory_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(IngestionError):
        discover_pdfs(tmp_path / "nope", max_size_bytes=MAX)


def test_size_limit_enforced(tmp_path: Path) -> None:
    pdf = make_pdf(tmp_path / "a.pdf", ["text"])
    with pytest.raises(PDFValidationError, match="size limit"):
        validate_pdf_file(pdf, max_size_bytes=10)
