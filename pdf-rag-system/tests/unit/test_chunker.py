import re

import pytest

from app.core.tokenizer import TiktokenTokenizer
from app.ingestion.chunker import TextChunker, split_sentences
from app.ingestion.models import PageText
from tests.helpers import WhitespaceTokenizer


def sentences(n: int, words: int = 10, prefix: str = "S") -> list[str]:
    """n distinct sentences of exactly `words` words each."""
    return [" ".join([f"{prefix}{i}w{j}" for j in range(words - 1)] + [f"end{i}."]).capitalize() for i in range(n)]


# --------------------------------------------------------------------------
# Sentence splitting
# --------------------------------------------------------------------------


def test_split_sentences_basic() -> None:
    assert split_sentences("First one. Second one! Third? Yes.") == ["First one.", "Second one!", "Third?", "Yes."]


def test_split_sentences_keeps_abbreviations_decimals_and_initials() -> None:
    text = "See Fig. 3 for details. Version 2.5 is used, e.g. In production. J. Smith wrote it."
    assert split_sentences(text) == [
        "See Fig. 3 for details.",
        "Version 2.5 is used, e.g. In production.",
        "J. Smith wrote it.",
    ]


def test_split_sentences_requires_capital_after_period() -> None:
    assert split_sentences("ends with a period. then lowercase continues") == [
        "ends with a period. then lowercase continues"
    ]


# --------------------------------------------------------------------------
# Chunk sizes and overlap
# --------------------------------------------------------------------------


def test_chunks_respect_chunk_size(tokenizer: WhitespaceTokenizer) -> None:
    chunker = TextChunker(tokenizer, chunk_size=50, chunk_overlap=10)
    chunks = chunker.split_text(" ".join(sentences(30)))

    assert len(chunks) > 1
    assert all(c.token_count <= 50 for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_consecutive_chunks_overlap_by_whole_sentences(tokenizer: WhitespaceTokenizer) -> None:
    chunker = TextChunker(tokenizer, chunk_size=50, chunk_overlap=20)
    all_sentences = sentences(30)
    chunks = chunker.split_text(" ".join(all_sentences))

    for prev, nxt in zip(chunks, chunks[1:]):
        prev_sents = split_sentences(prev.content)
        next_sents = split_sentences(nxt.content)
        shared = [s for s in next_sents if s in prev_sents]
        assert shared, "expected overlap between consecutive chunks"
        assert next_sents[: len(shared)] == prev_sents[-len(shared):]   # tail of prev == head of next
        assert sum(len(s.split()) for s in shared) <= 20                # overlap within budget


def test_zero_overlap_produces_disjoint_chunks(tokenizer: WhitespaceTokenizer) -> None:
    chunker = TextChunker(tokenizer, chunk_size=40, chunk_overlap=0)
    chunks = chunker.split_text(" ".join(sentences(20)))
    words = [w for c in chunks for w in c.content.split()]
    assert len(words) == len(set(words)) == 20 * 10


def test_sentences_are_not_split_unnecessarily(tokenizer: WhitespaceTokenizer) -> None:
    chunker = TextChunker(tokenizer, chunk_size=50, chunk_overlap=10)
    all_sentences = sentences(25)
    chunks = chunker.split_text(" ".join(all_sentences))

    for sentence in all_sentences:
        assert any(sentence in c.content for c in chunks), f"sentence was split: {sentence!r}"
    for c in chunks:
        assert c.content.endswith("."), "chunks should end on a sentence boundary"


def test_long_last_sentence_still_gets_token_overlap(tokenizer: WhitespaceTokenizer) -> None:
    chunker = TextChunker(tokenizer, chunk_size=50, chunk_overlap=10)
    # 30-word sentences exceed the 10-token overlap budget.
    chunks = chunker.split_text(" ".join(sentences(6, words=30)))
    for prev, nxt in zip(chunks, chunks[1:]):
        tail = prev.content.split()[-5:]
        assert " ".join(tail) in nxt.content


def test_oversized_sentence_is_split_on_word_boundaries(tokenizer: WhitespaceTokenizer) -> None:
    chunker = TextChunker(tokenizer, chunk_size=40, chunk_overlap=10)
    giant = " ".join(f"word{i}" for i in range(200)) + "."
    chunks = chunker.split_text(giant)

    assert len(chunks) >= 5
    assert all(c.token_count <= 40 for c in chunks)
    joined = " ".join(c.content for c in chunks).split()
    assert set(joined) == set(giant.split())  # nothing lost


def test_invalid_configuration_rejected(tokenizer: WhitespaceTokenizer) -> None:
    with pytest.raises(ValueError):
        TextChunker(tokenizer, chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValueError):
        TextChunker(tokenizer, chunk_size=0, chunk_overlap=0)


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------


def test_page_references_are_preserved_across_pages(tokenizer: WhitespaceTokenizer) -> None:
    chunker = TextChunker(tokenizer, chunk_size=50, chunk_overlap=10)
    # Every word encodes its page: "pg2x..." lives on page 2.
    pages = [PageText(n, " ".join(sentences(4, prefix=f"pg{n}x"))) for n in (1, 2, 3)]
    chunks = chunker.split_pages(pages)

    assert any(c.page_number != c.page_end for c in chunks), "expected a chunk spanning pages"
    for c in chunks:
        found = {int(p) for p in re.findall(r"pg(\d+)x", c.content, flags=re.IGNORECASE)}
        assert (min(found), max(found)) == (c.page_number, c.page_end)
    assert chunks[0].page_number == 1
    assert chunks[-1].page_end == 3


def test_empty_pages_produce_no_chunks(tokenizer: WhitespaceTokenizer) -> None:
    chunker = TextChunker(tokenizer, chunk_size=50, chunk_overlap=10)
    assert chunker.split_pages([PageText(1, ""), PageText(2, "   \n\n  ")]) == []

    chunks = chunker.split_pages([PageText(1, ""), PageText(2, "Only page two has text."), PageText(3, "")])
    assert len(chunks) == 1
    assert (chunks[0].page_number, chunks[0].page_end) == (2, 2)
    assert all(c.content.strip() for c in chunks)


def test_paragraph_breaks_are_kept(tokenizer: WhitespaceTokenizer) -> None:
    chunker = TextChunker(tokenizer, chunk_size=100, chunk_overlap=10)
    chunks = chunker.split_text("First paragraph here.\n\nSecond paragraph here.")
    assert chunks[0].content == "First paragraph here.\n\nSecond paragraph here."


def test_real_tokenizer_default_settings() -> None:
    tok = TiktokenTokenizer("cl100k_base")
    chunker = TextChunker(tok, chunk_size=800, chunk_overlap=150)
    text = " ".join(
        f"Component {i} of the retrieval pipeline converts questions into embeddings and queries PostgreSQL."
        for i in range(400)
    )
    chunks = chunker.split_text(text)
    assert len(chunks) > 3
    assert all(c.token_count <= 800 for c in chunks)
    assert all(c.token_count >= 500 for c in chunks[:-1])  # chunks are reasonably full
