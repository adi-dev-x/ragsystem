"""HTTP routes.

Handlers are plain ``def`` functions: the database driver and SDK clients are
synchronous, so FastAPI runs each request in its worker thread pool.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, status
from fastapi.responses import JSONResponse

from app.api.dependencies import get_container, get_document_reader, get_health_check, get_rag_service
from app.api.schemas import (
    DatabaseStatus,
    DocumentInfo,
    DocumentListResponse,
    ErrorResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SourceReference,
    Usage,
)
from app.container import DocumentReader, HealthCheck, ServiceContainer
from app.services.rag_service import RAGService
from app.vectorstore.models import SearchFilters

health_router = APIRouter(tags=["health"])
api_router = APIRouter(prefix="/api/v1")

_ERRORS = {
    422: {"model": ErrorResponse, "description": "Validation error or model refusal"},
    502: {"model": ErrorResponse, "description": "Embedding or LLM provider error"},
    503: {"model": ErrorResponse, "description": "Database unavailable"},
}


@health_router.get(
    "/health",
    response_model=HealthResponse,
    responses={503: {"model": HealthResponse, "description": "Database unreachable"}},
)
def health(
    checker: HealthCheck = Depends(get_health_check),
    container: ServiceContainer = Depends(get_container),
) -> JSONResponse:
    db = checker.check_health()
    body = HealthResponse(
        status="ok" if db.ok and db.schema_ready else "degraded",
        database=DatabaseStatus(
            status="ok" if db.ok else "error",
            latency_ms=db.latency_ms,
            pgvector_version=db.pgvector_version,
            schema_ready=db.schema_ready,
            error=db.error,
        ),
        llm_provider=container.settings.llm_provider,
        embedding_model=container.settings.embedding_model,
    )
    return JSONResponse(
        status_code=status.HTTP_200_OK if db.ok else status.HTTP_503_SERVICE_UNAVAILABLE,
        content=body.model_dump(mode="json", exclude_none=True),
    )


@api_router.get("/documents", response_model=DocumentListResponse, tags=["documents"], responses={503: _ERRORS[503]})
def list_documents(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    reader: DocumentReader = Depends(get_document_reader),
) -> DocumentListResponse:
    documents = reader.list_documents(limit=limit, offset=offset)
    return DocumentListResponse(
        total=reader.count_documents(),
        limit=limit,
        offset=offset,
        documents=[
            DocumentInfo(
                id=d.id,
                filename=d.filename,
                file_hash=d.file_hash,
                file_size=d.file_size,
                total_pages=d.total_pages,
                total_chunks=d.total_chunks,
                embedding_model=d.embedding_model,
                metadata=d.metadata,
                created_at=d.created_at,
                updated_at=d.updated_at,
            )
            for d in documents
        ],
    )


@api_router.post("/rag/query", response_model=QueryResponse, tags=["rag"], responses=_ERRORS)
def query(request: QueryRequest, service: RAGService = Depends(get_rag_service)) -> QueryResponse:
    filters = None
    if request.filters:
        f = request.filters
        filters = SearchFilters(
            filenames=tuple(f.filenames) if f.filenames else None,
            document_ids=tuple(f.document_ids) if f.document_ids else None,
            page_from=f.page_from,
            page_to=f.page_to,
        )

    result = service.answer(
        request.question,
        top_k=request.top_k,
        similarity_threshold=request.similarity_threshold,
        filters=filters,
    )
    return QueryResponse(
        question=result.question,
        answer=result.answer,
        sources=[
            SourceReference(
                id=number,
                document=s.filename,
                document_id=s.document_id,
                page=s.page_number,
                page_end=s.page_end,
                chunk_id=s.chunk_id,
                chunk_index=s.chunk_index,
                relevance_score=round(s.similarity, 4),
                content=s.content,
            )
            # Same order and numbering as the <source id="N"> tags the model cited.
            for number, s in enumerate(result.sources, start=1)
        ],
        model=result.model,
        provider=result.provider,
        usage=Usage(input_tokens=result.input_tokens, output_tokens=result.output_tokens),
    )
