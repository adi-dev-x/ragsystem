"""Orchestrates retrieval and grounded answer generation."""

from __future__ import annotations

import time
from dataclasses import dataclass

from app.core.logging import get_logger
from app.llm.base import LLMProvider
from app.llm.formatting import clean_answer
from app.llm.prompts import AnswerFormat, CitationStyle, build_system_prompt, build_user_prompt, not_found_answer
from app.retrieval.context_builder import ContextBuilder
from app.retrieval.retriever import Retriever
from app.vectorstore.models import ChunkSearchResult, SearchFilters

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RAGAnswer:
    question: str
    answer: str
    sources: list[ChunkSearchResult]
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    context_truncated: bool = False


class RAGService:
    def __init__(
        self,
        *,
        retriever: Retriever,
        context_builder: ContextBuilder,
        llm: LLMProvider,
        answer_format: AnswerFormat = "plain",
        citation_style: CitationStyle = "numbered",
    ) -> None:
        self._retriever = retriever
        self._context_builder = context_builder
        self._llm = llm
        self._answer_format: AnswerFormat = answer_format
        self._citation_style: CitationStyle = citation_style
        self._system_prompt = build_system_prompt(answer_format, citation_style)

    @property
    def provider_name(self) -> str:
        return self._llm.name

    @property
    def model_name(self) -> str:
        return self._llm.model

    def answer(
        self,
        question: str,
        *,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
        filters: SearchFilters | None = None,
    ) -> RAGAnswer:
        started = time.perf_counter()
        retrieval = self._retriever.retrieve(
            question, top_k=top_k, similarity_threshold=similarity_threshold, filters=filters
        )

        if not retrieval.chunks:
            # Nothing relevant enough: answer "not found" without calling the LLM,
            # which removes any chance of an ungrounded answer.
            logger.info("no relevant context; skipping generation", extra={"candidates": retrieval.candidates})
            return RAGAnswer(
                question=question,
                answer=not_found_answer(self._answer_format),
                sources=[],
                provider=self._llm.name,
                model=self._llm.model,
                input_tokens=0,
                output_tokens=0,
            )

        context = self._context_builder.build(retrieval.chunks)
        response = self._llm.generate(
            system_prompt=self._system_prompt,
            user_prompt=build_user_prompt(question, context.text, self._answer_format),
        )
        answer_text = clean_answer(
            response.text,
            answer_format=self._answer_format,
            citation_style=self._citation_style,
            source_count=len(context.chunks),
        )
        logger.info(
            "answer generated",
            extra={
                "provider": response.provider,
                "model": response.model,
                "sources": len(context.chunks),
                "context_tokens": context.token_count,
                "context_truncated": context.truncated,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "stop_reason": response.stop_reason,
                "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            },
        )
        return RAGAnswer(
            question=question,
            answer=answer_text or response.text,
            sources=context.chunks,
            provider=response.provider,
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            context_truncated=context.truncated,
        )
