"""Sentence-aware, token-budgeted chunking that keeps page references.

Text is broken into units (sentences, tagged with their page and whether they
start a paragraph). Units are packed greedily into chunks of at most
``chunk_size`` tokens. Consecutive chunks share roughly ``chunk_overlap``
tokens: the trailing whole sentences of the previous chunk are repeated. Only
when a single sentence is longer than the overlap budget is it cut, and only
sentences longer than ``chunk_size - chunk_overlap`` are split on word
boundaries. Chunks may span pages; each records its first and last page.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from app.core.tokenizer import Tokenizer
from app.ingestion.models import PageText, TextChunk

# Common abbreviations that end with a period but don't end a sentence.
_ABBREVIATIONS = frozenset(
    {
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e", "cf",
        "fig", "figs", "eq", "eqs", "no", "nos", "vol", "al", "inc", "ltd", "co", "corp",
        "approx", "dept", "est", "sec", "ch", "pp", "p", "ref", "refs", "u.s", "u.k", "jan",
        "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    }
)
_BOUNDARY_RE = re.compile(r"([.!?]+)([\"'”’)\]]*)(\s+)")
_SENTENCE_START_CHARS = "\"'“‘(["


def split_sentences(text: str) -> list[str]:
    """Split a paragraph into sentences with a conservative rule-based splitter."""
    sentences: list[str] = []
    start = 0
    for match in _BOUNDARY_RE.finditer(text):
        next_char = text[match.end(): match.end() + 1]
        if not next_char or not (next_char.isupper() or next_char.isdigit() or next_char in _SENTENCE_START_CHARS):
            continue
        if match.group(1) == ".":
            preceding = text[start: match.start()].split()
            last_word = preceding[-1] if preceding else ""
            if last_word.lower().strip("(\"'") in _ABBREVIATIONS:
                continue
            if len(last_word) == 1 and last_word.isupper():  # initials, e.g. "J. Smith"
                continue
        end = match.start() + len(match.group(1)) + len(match.group(2))
        sentence = text[start:end].strip()
        if sentence:
            sentences.append(sentence)
        start = match.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


@dataclass(frozen=True, slots=True)
class _Unit:
    text: str
    page: int
    tokens: int
    paragraph_start: bool


class TextChunker:
    def __init__(self, tokenizer: Tokenizer, *, chunk_size: int = 800, chunk_overlap: int = 150) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
        self._tok = tokenizer
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Longest unit that still fits next to a full overlap tail.
        self._max_unit_tokens = chunk_size - chunk_overlap

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split_pages(self, pages: Sequence[PageText]) -> list[TextChunk]:
        """Chunk a document's pages. Empty pages contribute nothing."""
        return self._pack(self._to_units(pages))

    def split_text(self, text: str, page_number: int = 1) -> list[TextChunk]:
        return self.split_pages([PageText(page_number=page_number, text=text)])

    # ------------------------------------------------------------------
    # Units
    # ------------------------------------------------------------------

    def _to_units(self, pages: Sequence[PageText]) -> list[_Unit]:
        units: list[_Unit] = []
        for page in pages:
            if page.is_empty:
                continue
            for paragraph in page.text.split("\n\n"):
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                first = True
                for sentence in split_sentences(paragraph):
                    for piece in self._split_oversized(sentence):
                        units.append(_Unit(piece, page.page_number, self._tok.count(piece), first))
                        first = False
        return units

    def _split_oversized(self, text: str) -> list[str]:
        """Split a sentence that can't fit in a chunk alongside the overlap, on word boundaries."""
        if self._tok.count(text) <= self._max_unit_tokens:
            return [text]
        limit = self._max_unit_tokens
        pieces: list[str] = []
        words: list[str] = []
        words_tokens = 0

        def flush() -> None:
            nonlocal words, words_tokens
            if words:
                pieces.append(" ".join(words))
            words, words_tokens = [], 0

        for word in text.split():
            word_tokens = self._tok.count(" " + word)
            if word_tokens > limit:
                # A single enormous "word" (URL, base64 blob): cut it by tokens.
                flush()
                ids = self._tok.encode(word)
                for i in range(0, len(ids), limit):
                    piece = self._tok.decode(ids[i: i + limit]).strip()
                    if piece:
                        pieces.append(piece)
                continue
            if words and words_tokens + word_tokens > limit:
                flush()
            words.append(word)
            words_tokens += word_tokens
        flush()
        return pieces

    # ------------------------------------------------------------------
    # Packing
    # ------------------------------------------------------------------

    @staticmethod
    def _cost(units: Sequence[_Unit]) -> int:
        # One token per separator keeps the estimate conservative.
        return sum(u.tokens for u in units) + max(0, len(units) - 1)

    def _pack(self, units: Sequence[_Unit]) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        current: list[_Unit] = []
        has_new_content = False

        for unit in units:
            if current and self._cost(current) + 1 + unit.tokens > self.chunk_size:
                self._emit(chunks, current)
                current = self._overlap_tail(current)
                # Trim the overlap if it would push the next unit over budget.
                while current and self._cost(current) + 1 + unit.tokens > self.chunk_size:
                    current.pop(0)
            current.append(unit)
            has_new_content = True

        if current and has_new_content:
            self._emit(chunks, current)
        return chunks

    def _overlap_tail(self, units: Sequence[_Unit]) -> list[_Unit]:
        if self.chunk_overlap == 0 or not units:
            return []
        tail: list[_Unit] = []
        for unit in reversed(units):
            if self._cost([unit, *tail]) > self.chunk_overlap:
                break
            tail.insert(0, unit)
        if tail:
            return tail

        # The last sentence alone exceeds the overlap: reuse its final tokens.
        last = units[-1]
        ids = self._tok.encode(last.text)
        text = self._tok.decode(ids[-self.chunk_overlap:]).replace("�", "")
        if text and not text[0].isspace() and len(ids) > self.chunk_overlap:
            # Drop the leading partial word.
            _, _, text = text.partition(" ")
        text = text.strip()
        if not text:
            return []
        return [_Unit(text, last.page, self._tok.count(text), paragraph_start=False)]

    def _emit(self, chunks: list[TextChunk], units: Sequence[_Unit]) -> None:
        parts: list[str] = []
        for i, unit in enumerate(units):
            if i:
                parts.append("\n\n" if unit.paragraph_start else " ")
            parts.append(unit.text)
        content = "".join(parts).strip()
        if not content:
            return
        chunks.append(
            TextChunk(
                chunk_index=len(chunks),
                content=content,
                page_number=min(u.page for u in units),
                page_end=max(u.page for u in units),
                token_count=self._tok.count(content),
            )
        )
