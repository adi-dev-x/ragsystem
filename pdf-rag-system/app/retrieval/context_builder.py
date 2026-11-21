"""Turn retrieved chunks into a bounded, citable context block for the LLM."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from app.core.tokenizer import Tokenizer
from app.vectorstore.models import ChunkSearchResult

# Neutralise anything in document text that could close our wrapper tags.
_TAG_RE = re.compile(r"</?\s*(source|context|question)\b[^>]*>", re.IGNORECASE)


def page_label(page_number: int, page_end: int) -> str:
    return f"p. {page_number}" if page_number == page_end else f"pp. {page_number}-{page_end}"


def _escape_attr(value: str) -> str:
    return value.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")


@dataclass(frozen=True, slots=True)
class BuiltContext:
    text: str
    chunks: list[ChunkSearchResult]   # chunks actually included, in context order
    token_count: int
    truncated: bool                    # True if any retrieved chunk was cut or left out


class ContextBuilder:
    def __init__(self, tokenizer: Tokenizer, *, max_tokens: int = 6000) -> None:
        self._tok = tokenizer
        self.max_tokens = max_tokens

    def build(self, chunks: Sequence[ChunkSearchResult]) -> BuiltContext:
        """Add chunks in relevance order until the token budget is spent.

        The most relevant chunk is always included (cut to fit if necessary);
        later chunks that don't fit are skipped.
        """
        blocks: list[str] = []
        used: list[ChunkSearchResult] = []
        total = 0
        truncated = False

        for chunk in chunks:
            header = (
                f'<source id="{len(used) + 1}" document="{_escape_attr(chunk.filename)}" '
                f'pages="{page_label(chunk.page_number, chunk.page_end)}">'
            )
            body = _TAG_RE.sub(lambda m: m.group(0).replace("<", "&lt;"), chunk.content.strip())
            block = f"{header}\n{body}\n</source>"
            cost = self._tok.count(block)

            if total + cost > self.max_tokens:
                truncated = True
                if used:
                    continue
                # Nothing fits yet: include a trimmed version of the best chunk.
                overhead = self._tok.count(f"{header}\n\n</source>")
                budget = max(0, self.max_tokens - overhead)
                body = self._tok.decode(self._tok.encode(body)[:budget]).strip()
                if not body:
                    break
                block = f"{header}\n{body}\n</source>"
                cost = self._tok.count(block)

            blocks.append(block)
            used.append(chunk)
            total += cost

        return BuiltContext(text="\n\n".join(blocks), chunks=used, token_count=total, truncated=truncated)
