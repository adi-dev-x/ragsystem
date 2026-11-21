"""Wire concrete services together (composition root for the API)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.config import Settings
from app.core.exceptions import ConfigurationError, DatabaseError
from app.core.logging import get_logger
from app.core.tokenizer import TiktokenTokenizer
from app.embeddings.embedding_service import create_embedding_service
from app.llm.factory import create_llm_provider
from app.retrieval.context_builder import ContextBuilder
from app.retrieval.retriever import Retriever
from app.services.rag_service import RAGService
from app.vectorstore.database import Database
from app.vectorstore.models import DatabaseHealth, DocumentRecord
from app.vectorstore.repository import DocumentRepository

logger = get_logger(__name__)


class HealthCheck(Protocol):
    def check_health(self) -> DatabaseHealth: ...
    def close(self) -> None: ...


class DocumentReader(Protocol):
    def list_documents(self, *, limit: int = 100, offset: int = 0) -> list[DocumentRecord]: ...
    def count_documents(self) -> int: ...


@dataclass
class ServiceContainer:
    settings: Settings
    database: HealthCheck
    documents: DocumentReader
    rag_service: RAGService

    def close(self) -> None:
        self.database.close()


def build_container(settings: Settings) -> ServiceContainer:
    database = Database(settings, application_name="pdf-rag-api")
    # Don't block startup on the database: /health reports its status.
    database.open(wait=False)
    try:
        repository = DocumentRepository(database)
        _check_embedding_dimension(repository, settings)

        tokenizer = TiktokenTokenizer(settings.tokenizer_encoding)
        retriever = Retriever(
            embedding_service=create_embedding_service(settings),
            searcher=repository,
            default_top_k=settings.top_k,
            similarity_threshold=settings.similarity_threshold,
            candidate_multiplier=settings.retrieval_candidate_multiplier,
            ef_search=settings.hnsw_ef_search,
        )
        rag_service = RAGService(
            retriever=retriever,
            context_builder=ContextBuilder(tokenizer, max_tokens=settings.max_context_tokens),
            llm=create_llm_provider(settings),
            answer_format=settings.answer_format,
            citation_style=settings.citation_style,
        )
    except Exception:
        database.close()
        raise
    return ServiceContainer(settings=settings, database=database, documents=repository, rag_service=rag_service)


def _check_embedding_dimension(repository: DocumentRepository, settings: Settings) -> None:
    try:
        dims = repository.get_embedding_dimension()
    except DatabaseError as exc:
        logger.warning("could not verify embedding dimension at startup", extra={"error": exc.message})
        return
    if dims is None:
        logger.warning("document_chunks table not found; run `python -m scripts.migrate`")
    elif dims != settings.embedding_dimensions:
        raise ConfigurationError(
            f"EMBEDDING_DIMENSIONS={settings.embedding_dimensions} does not match "
            f"document_chunks.embedding vector({dims}); run "
            f"`python -m scripts.migrate --set-embedding-dimensions {settings.embedding_dimensions}`"
        )
