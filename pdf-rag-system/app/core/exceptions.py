"""Domain exceptions.

Every exception carries a stable machine-readable ``code`` and an HTTP status
so the API layer can map failures to structured responses without leaking
internals. Messages must never contain secrets.
"""

from __future__ import annotations

from typing import Any


class RAGError(Exception):
    code: str = "internal_error"
    status_code: int = 500

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(RAGError):
    code = "configuration_error"
    status_code = 500


# --- Ingestion -------------------------------------------------------------


class IngestionError(RAGError):
    code = "ingestion_error"


class PDFValidationError(IngestionError):
    code = "invalid_pdf"
    status_code = 422


class PDFExtractionError(IngestionError):
    code = "pdf_extraction_failed"
    status_code = 422


class IngestionLockError(IngestionError):
    code = "ingestion_in_progress"
    status_code = 409


# --- Database --------------------------------------------------------------


class DatabaseError(RAGError):
    code = "database_error"
    status_code = 500


class DatabaseConnectionError(DatabaseError):
    code = "database_unavailable"
    status_code = 503


class MigrationError(DatabaseError):
    code = "migration_failed"


# --- External services -----------------------------------------------------


class ExternalServiceError(RAGError):
    """Failure talking to an upstream API.

    ``retryable`` marks transient failures (rate limits, timeouts, 5xx) that the
    retry helper may repeat; ``retry_after`` carries a server-provided delay.
    """

    code = "upstream_error"
    status_code = 502

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        retry_after: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details=details)
        self.retryable = retryable
        self.retry_after = retry_after


class EmbeddingError(ExternalServiceError):
    code = "embedding_error"


class EmbeddingDimensionError(EmbeddingError):
    code = "embedding_dimension_mismatch"
    status_code = 500


class LLMError(ExternalServiceError):
    code = "llm_error"


class LLMRefusalError(LLMError):
    code = "llm_refusal"
    status_code = 422


# --- Retrieval -------------------------------------------------------------


class RetrievalError(RAGError):
    code = "retrieval_error"
