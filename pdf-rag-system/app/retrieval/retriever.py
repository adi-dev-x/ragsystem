"""Vector retrieval, kept independent of answer generation.

Flow: embed query -> over-fetch nearest chunks from pgvector -> apply the
similarity threshold -> drop duplicate content -> optional rerank -> top_k.

Extension points: pass a ``reranker`` (e.g. a cross-encoder) to reorder
candidates, or replace ``search`` with a hybrid (vector + full-text) searcher
that returns the same :class:`ChunkSearchResult` records.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from app.core.logging import get_logger
from app.embeddings.embedding_service import EmbeddingService
from app.vectorstore.models import ChunkSearchResult, SearchFilters

logger = get_logger(__name__)


class ChunkSearcher(Protocol):
    def search_similar(
        self,
        embedding: Sequence[float],
        *,
        limit: int,
        filters: SearchFilters | None = None,
        ef_search: int = 100,
    ) -> list[ChunkSearchResult]: ...


class Reranker(Protocol):
    def rerank(self, query: str, chunks: list[ChunkSearchResult]) -> list[ChunkSearchResult]: ...


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    chunks: list[ChunkSearchResult]
    candidates: int            # rows returned by the vector search
    below_threshold: int       # candidates dropped by the similarity threshold
    duplicates_removed: int
    threshold: float
    duration_ms: float


def _content_key(text: str) -> str:
    return " ".join(text.split()).casefold()


def deduplicate(chunks: Sequence[ChunkSearchResult]) -> list[ChunkSearchResult]:
    """Keep the best-scoring copy of identical chunk text (input is sorted by score)."""
    seen_ids: set[object] = set()
    seen_text: set[str] = set()
    unique: list[ChunkSearchResult] = []
    for chunk in chunks:
        key = _content_key(chunk.content)
        if chunk.chunk_id in seen_ids or key in seen_text:
            continue
        seen_ids.add(chunk.chunk_id)
        seen_text.add(key)
        unique.append(chunk)
    return unique


class Retriever:
    def __init__(
        self,
        *,
        embedding_service: EmbeddingService,
        searcher: ChunkSearcher,
        default_top_k: int = 5,
        similarity_threshold: float = 0.3,
        candidate_multiplier: int = 3,
        ef_search: int = 100,
        reranker: Reranker | None = None,
    ) -> None:
        self._embedder = embedding_service
        self._searcher = searcher
        self.default_top_k = default_top_k
        self.similarity_threshold = similarity_threshold
        self._multiplier = max(1, candidate_multiplier)
        self._ef_search = ef_search
        self._reranker = reranker

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
        filters: SearchFilters | None = None,
    ) -> RetrievalResult:
        started = time.perf_counter()
        k = top_k or self.default_top_k
        threshold = self.similarity_threshold if similarity_threshold is None else similarity_threshold

        query_vector = self._embedder.embed_query(query)
        candidates = self._searcher.search_similar(
            query_vector,
            limit=k * self._multiplier,
            filters=filters if filters and not filters.is_empty() else None,
            ef_search=self._ef_search,
        )
        relevant = [c for c in candidates if c.similarity >= threshold]
        unique = deduplicate(relevant)
        if self._reranker is not None and unique:
            unique = self._reranker.rerank(query, unique)
        selected = unique[:k]

        result = RetrievalResult(
            chunks=selected,
            candidates=len(candidates),
            below_threshold=len(candidates) - len(relevant),
            duplicates_removed=len(relevant) - len(unique),
            threshold=threshold,
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        logger.info(
            "retrieval complete",
            extra={
                "top_k": k,
                "threshold": threshold,
                "candidates": result.candidates,
                "below_threshold": result.below_threshold,
                "duplicates_removed": result.duplicates_removed,
                "returned": len(selected),
                "best_score": round(selected[0].similarity, 4) if selected else None,
                "duration_ms": result.duration_ms,
            },
        )
        return result
