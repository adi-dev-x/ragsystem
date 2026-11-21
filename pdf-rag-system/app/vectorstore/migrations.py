"""Minimal forward-only SQL migration runner.

Migrations are plain ``NNN_description.sql`` files in ``migrations/``. Each is
applied once, inside its own transaction, and recorded in
``schema_migrations`` together with a checksum so edits to an already-applied
file are detected. A PostgreSQL advisory lock serialises concurrent runners.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import psycopg

from app.config import PROJECT_ROOT
from app.core.exceptions import DatabaseConnectionError, MigrationError
from app.core.logging import get_logger

logger = get_logger(__name__)

MIGRATIONS_DIR = PROJECT_ROOT / "migrations"
_MIGRATION_LOCK_ID = 7_310_001  # arbitrary, constant advisory-lock key
_FILENAME_RE = re.compile(r"^(\d+)_[\w\-]+\.sql$")


@dataclass(frozen=True, slots=True)
class Migration:
    version: str
    name: str
    path: Path
    sql: str

    @property
    def checksum(self) -> str:
        return hashlib.sha256(self.sql.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class MigrationStatus:
    version: str
    name: str
    applied: bool


def discover_migrations(directory: Path = MIGRATIONS_DIR) -> list[Migration]:
    if not directory.is_dir():
        raise MigrationError(f"Migrations directory not found: {directory}")
    migrations: list[Migration] = []
    for path in sorted(directory.glob("*.sql")):
        match = _FILENAME_RE.match(path.name)
        if not match:
            logger.warning("ignoring migration with unexpected name", extra={"file": path.name})
            continue
        migrations.append(Migration(match.group(1), path.stem, path, path.read_text(encoding="utf-8")))
    versions = [m.version for m in migrations]
    if len(versions) != len(set(versions)):
        raise MigrationError("Duplicate migration version numbers found")
    return migrations


def _connect(database_url: str) -> psycopg.Connection:
    try:
        return psycopg.connect(database_url, autocommit=True, connect_timeout=10, application_name="pdf-rag-migrate")
    except psycopg.OperationalError as exc:
        raise DatabaseConnectionError("Could not connect to PostgreSQL to run migrations") from exc


def _ensure_table(conn: psycopg.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version    TEXT PRIMARY KEY,
            name       TEXT NOT NULL,
            checksum   TEXT NOT NULL,
            applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )


def apply_migrations(database_url: str, directory: Path = MIGRATIONS_DIR) -> list[str]:
    """Apply pending migrations. Returns the versions applied in this call."""
    migrations = discover_migrations(directory)
    applied_now: list[str] = []
    with _connect(database_url) as conn:
        conn.execute("SELECT pg_advisory_lock(%s)", [_MIGRATION_LOCK_ID])
        try:
            _ensure_table(conn)
            rows = conn.execute("SELECT version, checksum FROM schema_migrations").fetchall()
            applied = {version: checksum for version, checksum in rows}

            for migration in migrations:
                if migration.version in applied:
                    if applied[migration.version] != migration.checksum:
                        logger.warning(
                            "applied migration file has changed since it was run",
                            extra={"migration": migration.name},
                        )
                    continue
                logger.info("applying migration", extra={"migration": migration.name})
                try:
                    with conn.transaction():
                        conn.execute(migration.sql)  # type: ignore[arg-type]  # trusted, multi-statement
                        conn.execute(
                            "INSERT INTO schema_migrations (version, name, checksum) VALUES (%s, %s, %s)",
                            [migration.version, migration.name, migration.checksum],
                        )
                except psycopg.Error as exc:
                    raise MigrationError(f"Migration {migration.name} failed: {exc}") from exc
                applied_now.append(migration.version)
        finally:
            conn.execute("SELECT pg_advisory_unlock(%s)", [_MIGRATION_LOCK_ID])

    if applied_now:
        logger.info("migrations applied", extra={"versions": applied_now})
    else:
        logger.info("database schema is up to date")
    return applied_now


def migration_status(database_url: str, directory: Path = MIGRATIONS_DIR) -> list[MigrationStatus]:
    migrations = discover_migrations(directory)
    with _connect(database_url) as conn:
        _ensure_table(conn)
        applied = {row[0] for row in conn.execute("SELECT version FROM schema_migrations").fetchall()}
    return [MigrationStatus(m.version, m.name, m.version in applied) for m in migrations]


MAX_HNSW_DIMENSIONS = 2000  # pgvector's HNSW limit for the `vector` type


def count_indexed_documents(database_url: str) -> int:
    with _connect(database_url) as conn:
        row = conn.execute("SELECT count(*) FROM documents").fetchone()
    return int(row[0]) if row else 0


def resize_embedding_dimension(database_url: str, dimensions: int) -> int:
    """Change ``document_chunks.embedding`` to ``vector(dimensions)``.

    Vectors of different sizes can't be compared, so every indexed document is
    deleted (re-run ingestion afterwards). Returns the number of documents
    removed. Runs in one transaction with the HNSW index rebuilt.
    """
    if not 1 <= dimensions <= MAX_HNSW_DIMENSIONS:
        raise MigrationError(f"Embedding dimensions must be between 1 and {MAX_HNSW_DIMENSIONS} for an HNSW index")
    from psycopg import sql

    with _connect(database_url) as conn:
        try:
            with conn.transaction():
                conn.execute("LOCK TABLE documents, document_chunks IN ACCESS EXCLUSIVE MODE")
                removed = conn.execute("DELETE FROM documents").rowcount  # chunks cascade
                conn.execute("DROP INDEX IF EXISTS document_chunks_embedding_hnsw_idx")
                conn.execute(
                    sql.SQL("ALTER TABLE document_chunks ALTER COLUMN embedding TYPE vector({})").format(
                        sql.Literal(dimensions)
                    )
                )
                conn.execute(
                    "CREATE INDEX document_chunks_embedding_hnsw_idx ON document_chunks "
                    "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
                )
        except psycopg.Error as exc:
            raise MigrationError(f"Could not change embedding dimensions: {exc}") from exc
    logger.info("embedding column resized", extra={"dimensions": dimensions, "documents_removed": removed})
    return removed
