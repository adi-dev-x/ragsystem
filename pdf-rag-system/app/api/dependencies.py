"""FastAPI dependencies resolving services from the application container."""

from __future__ import annotations

from fastapi import Request

from app.container import DocumentReader, HealthCheck, ServiceContainer
from app.services.rag_service import RAGService


def get_container(request: Request) -> ServiceContainer:
    return request.app.state.container


def get_rag_service(request: Request) -> RAGService:
    return get_container(request).rag_service


def get_document_reader(request: Request) -> DocumentReader:
    return get_container(request).documents


def get_health_check(request: Request) -> HealthCheck:
    return get_container(request).database
