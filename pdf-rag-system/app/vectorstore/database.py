"""PostgreSQL connection pool management."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool, PoolTimeout

from app.config import Settings
from app.core.exceptions import DatabaseConnectionError, DatabaseError
from app.core.logging import get_logger, mask_url
from app.vectorstore.models import DatabaseHealth

logger = get_logger(__name__)


class Database:
    """Thin wrapper over a psycopg connection pool.

    Connections are handed out by :meth:`connection`; leaving the context
    commits, an exception rolls back. Rows are returned as dicts.
    """

    def __init__(self, settings: Settings, *, application_name: str = "pdf-rag") -> None:
        self._url = settings.database_url.get_secret_value()
        self._timeout = settings.db_pool_timeout_seconds
        connect_kwargs: dict[str, object] = {
            "application_name": application_name,
            "connect_timeout": 10,
            "row_factory": dict_row,
        }
        if settings.db_statement_timeout_ms:
            connect_kwargs["options"] = f"-c statement_timeout={settings.db_statement_timeout_ms}"
        self._pool = ConnectionPool(
            conninfo=self._url,
            min_size=settings.db_pool_min_size,
            max_size=settings.db_pool_max_size,
            timeout=settings.db_pool_timeout_seconds,
            kwargs=connect_kwargs,
            check=ConnectionPool.check_connection,
            open=False,
            name=application_name,
        )

    # -- lifecycle ---------------------------------------------------------

    def open(self, *, wait: bool = False) -> None:
        """Open the pool. ``wait=True`` blocks until ``min_size`` connections exist."""
        logger.info("opening database pool", extra={"database": mask_url(self._url), "wait": wait})
        try:
            self._pool.open(wait=wait, timeout=self._timeout)
        except PoolTimeout as exc:
            raise DatabaseConnectionError("Timed out connecting to PostgreSQL") from exc

    def close(self) -> None:
        self._pool.close()

    def __enter__(self) -> "Database":
        self.open(wait=True)
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- connections -------------------------------------------------------

    @contextmanager
    def connection(self, timeout: float | None = None) -> Iterator[psycopg.Connection]:
        try:
            with self._pool.connection(timeout=timeout) as conn:
                yield conn
        except PoolTimeout as exc:
            raise DatabaseConnectionError("No database connection available (pool timeout)") from exc
        except psycopg.OperationalError as exc:
            raise DatabaseConnectionError("Lost connection to PostgreSQL") from exc
        except psycopg.Error as exc:
            raise DatabaseError(f"Database operation failed: {exc.__class__.__name__}: {exc}") from exc

    @contextmanager
    def transaction(self) -> Iterator[psycopg.Connection]:
        """Connection with an explicit transaction block (commit or roll back as a unit)."""
        with self.connection() as conn:
            with conn.transaction():
                yield conn

    # -- health ------------------------------------------------------------

    def check_health(self) -> DatabaseHealth:
        start = time.perf_counter()
        try:
            with self.connection(timeout=3) as conn:
                conn.execute("SELECT 1")
                row = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
                tables = conn.execute(
                    "SELECT to_regclass('public.documents') IS NOT NULL AS docs,"
                    " to_regclass('public.document_chunks') IS NOT NULL AS chunks"
                ).fetchone()
            latency = (time.perf_counter() - start) * 1000
            return DatabaseHealth(
                ok=True,
                latency_ms=round(latency, 2),
                pgvector_version=row["extversion"] if row else None,
                schema_ready=bool(tables and tables["docs"] and tables["chunks"]),
            )
        except DatabaseError as exc:
            logger.warning("database health check failed", extra={"error": exc.message})
            return DatabaseHealth(ok=False, error=exc.message)
