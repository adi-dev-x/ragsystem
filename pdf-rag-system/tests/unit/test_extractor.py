from pathlib import Path

import pytest

from app.core.exceptions import PDFValidationError
from app.ingestion.extractor import clean_block_text, extract_pdf
from tests.helpers import make_pdf


def test_extracts_text_page_by_page_with_page_numbers(tmp_path: Path) -> None:
    pdf = make_pdf(
        tmp_path / "guide.pdf",
        ["The ingestion server reads PDFs.", "The retrieval server answers questions."],
        title="System Guide",
    )
    doc = extract_pdf(pdf)

    assert doc.total_pages == 2
    assert [p.page_number for p in doc.pages] == [1, 2]
    assert "ingestion server" in doc.pages[0].text
    assert "retrieval server" in doc.pages[1].text
    assert doc.metadata["title"] == "System Guide"


def test_empty_page_is_kept_as_empty(tmp_path: Path) -> None:
    pdf = make_pdf(tmp_path / "gap.pdf", ["First page text.", "", "Third page text."])
    doc = extract_pdf(pdf)

    assert doc.total_pages == 3
    assert doc.empty_pages == [2]
    assert doc.pages[1].is_empty
    assert doc.has_text


def test_pdf_without_any_text(tmp_path: Path) -> None:
    pdf = make_pdf(tmp_path / "blank.pdf", ["", ""])
    doc = extract_pdf(pdf)

    assert doc.total_pages == 2
    assert not doc.has_text
    assert doc.empty_pages == [1, 2]


def test_corrupt_pdf_raises_validation_error(tmp_path: Path) -> None:
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"%PDF-1.7\nthis is not really a pdf")
    with pytest.raises(PDFValidationError):
        extract_pdf(bad)


def test_paragraphs_are_separated_and_lines_joined(tmp_path: Path) -> None:
    pdf = make_pdf(tmp_path / "wrap.pdf", ["line one of a paragraph\ncontinues here"])
    text = extract_pdf(pdf).pages[0].text
    assert "\n" not in text.strip() or "\n\n" in text
    assert "line one of a paragraph continues here" in text


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("exam-\nple text", "example text"),
        ("soft\nwrapped   line", "soft wrapped line"),
        ("null\x00byte", "nullbyte"),
        ("ﬁnance", "finance"),  # ligature normalised by NFKC
        ("  padded \t ", "padded"),
    ],
)
def test_clean_block_text(raw: str, expected: str) -> None:
    assert clean_block_text(raw) == expected
