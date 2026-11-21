"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.errors import register_exception_handlers
from app.api.middleware import RequestContextMiddleware
from app.api.routes import api_router, health_router
from app.config import Settings, get_settings
from app.container import ServiceContainer, build_container
from app.core.logging import configure_logging, get_logger

logger = get_logger(__name__)


def create_app(
    settings: Settings | None = None,
    *,
    container_factory: Callable[[Settings], ServiceContainer] = build_container,
    configure_logs: bool = True,
) -> FastAPI:
    """Build the API. Tests pass a ``container_factory`` that returns fakes."""
    settings = settings or get_settings()
    if configure_logs:
        configure_logging(settings.log_level, settings.log_format)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        container = container_factory(settings)
        app.state.container = container
        logger.info(
            "API started",
            extra={
                "llm_provider": settings.llm_provider,
                "llm_model": settings.llm_model,
                "embedding_model": settings.embedding_model,
            },
        )
        try:
            yield
        finally:
            container.close()
            logger.info("API stopped")

    app = FastAPI(
        title="PDF RAG API",
        version="1.0.0",
        description="Retrieval-augmented question answering over PDF documents stored in PostgreSQL/pgvector.",
        lifespan=lifespan,
    )
    app.add_middleware(RequestContextMiddleware)
    register_exception_handlers(app)
    app.include_router(health_router)
    app.include_router(api_router)
    return app
