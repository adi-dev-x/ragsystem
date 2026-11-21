"""Map exceptions to structured JSON error responses."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.schemas import ErrorBody, ErrorResponse
from app.core.exceptions import RAGError
from app.core.logging import get_logger, request_id_var

logger = get_logger(__name__)


def error_response(status_code: int, code: str, message: str, details: Any = None) -> JSONResponse:
    body = ErrorResponse(
        error=ErrorBody(code=code, message=message, details=details),
        request_id=request_id_var.get(),
    )
    return JSONResponse(status_code=status_code, content=jsonable_encoder(body, exclude_none=True))


async def _rag_error_handler(request: Request, exc: RAGError) -> JSONResponse:
    log = logger.error if exc.status_code >= 500 else logger.warning
    log("request failed", extra={"code": exc.code, "error": exc.message, "path": request.url.path})
    # Configuration and internal database details stay in the logs.
    if exc.status_code == 500:
        message = "The server is misconfigured." if exc.code == "configuration_error" else "Internal server error."
        return error_response(500, exc.code, message)
    return error_response(exc.status_code, exc.code, exc.message, exc.details or None)


async def _validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    details = [
        {"field": ".".join(str(p) for p in err.get("loc", ()) if p != "body"), "message": err.get("msg"), "type": err.get("type")}
        for err in exc.errors()
    ]
    return error_response(422, "validation_error", "Request validation failed.", details)


async def _http_error_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
    return error_response(exc.status_code, code, str(exc.detail))


async def _unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled error", extra={"path": request.url.path})
    return error_response(500, "internal_error", "Internal server error.")


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(RAGError, _rag_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, _validation_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(StarletteHTTPException, _http_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, _unhandled_error_handler)
