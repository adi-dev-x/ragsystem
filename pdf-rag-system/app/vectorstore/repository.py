"""Repository: every SQL statement the application runs lives here.

All queries are parameterised. Multi-statement writes run in a single
transaction so a document is never visible with a partial set of chunks.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any
from uuid import UUID

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

from app.core.exceptions import DatabaseError, IngestionLockError
from app.core.logging import get_logger
from app.vectorstore.database import Database
from app.vectorstore.models import (
    ChunkSearchResult,
    DocumentRecord,
    NewChunk,
    NewDocument,
    SearchFilters,
)

logger = get_logger(__name__)

_INGESTION_LOCK_ID = 7_310_002  # advisory-lock key; distinct from the migration lock

_DOCUMENT_COLUMNS = (
    "id, filename, file_hash, file_size, total_pages, total_chunks, embedding_model, "
    "chunk_size, chunk_overlap, metadata, created_at, updated_at"
)

_INSERT_CHUNK_SQL = """
    INSERT INTO document_chunks
        (document_id, chunk_index, content, page_number, page_end, token_count, embedding)
    VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
"""


def _to_document(row: dict[str, Any]) -> DocumentRecord:
    return DocumentRecord(**row)


class DocumentRepository:
    def __init__(self, database: Database) -> None:
        self._db = database
        self._iterative_scan_supported: bool | None = None

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def list_documents(self, *, limit: int = 100, offset: int = 0) -> list[DocumentRecord]:
        query = f"SELECT {_DOCUMENT_COLUMNS} FROM documents ORDER BY filename LIMIT %s OFFSET %s"
        with self._db.connection() as conn:
            rows = conn.execute(query, [limit, offset]).fetchall()
        return [_to_document(r) for r in rows]

    def list_all_documents(self) -> list[DocumentRecord]:
        with self._db.connection() as conn:
            rows = conn.execute(f"SELECT {_DOCUMENT_COLUMNS} FROM documents ORDER BY filename").fetchall()
        return [_to_document(r) for r in rows]

    def count_documents(self) -> int:
        with self._db.connection() as conn:
            row = conn.execute("SELECT count(*) AS n FROM documents").fetchone()
        return int(row["n"]) if row else 0

    def count_chunks(self, document_id: UUID | None = None) -> int:
        with self._db.connection() as conn:
            if document_id is None:
                row = conn.execute("SELECT count(*) AS n FROM document_chunks").fetchone()
            else:
                row = conn.execute(
                    "SELECT count(*) AS n FROM document_chunks WHERE document_id = %s", [document_id]
                ).fetchone()
        return int(row["n"]) if row else 0

    def get_embedding_dimension(self) -> int | None:
        """Dimension declared on ``document_chunks.embedding``; ``None`` if the table is missing."""
        query = """
            SELECT a.atttypmod AS dims
            FROM pg_attribute a
            WHERE a.attrelid = to_regclass('public.document_chunks')
              AND a.attname = 'embedding'
              AND NOT a.attisdropped
        """
        with self._db.connection() as conn:
            row = conn.execute(query).fetchone()
        if not row or row["dims"] is None or row["dims"] < 0:
            return None
        return int(row["dims"])

    # ------------------------------------------------------------------
    # Writes (each method is one transaction)
    # ------------------------------------------------------------------

    def insert_document(self, document: NewDocument, chunks: Sequence[NewChunk]) -> UUID:
        with self._db.transaction() as conn:
            row = conn.execute(
                """
                INSERT INTO documents
                    (filename, file_hash, file_size, total_pages, total_chunks,
                     embedding_model, chunk_size, chunk_overlap, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                [
                    document.filename,
                    document.file_hash,
                    document.file_size,
                    document.total_pages,
                    len(chunks),
                    document.embedding_model,
                    document.chunk_size,
                    document.chunk_overlap,
                    Jsonb(document.metadata),
                ],
            ).fetchone()
            document_id: UUID = row["id"]
            self._insert_chunks(conn, document_id, chunks)
        return document_id

    def replace_document(self, document_id: UUID, document: NewDocument, chunks: Sequence[NewChunk]) -> None:
        """Atomically swap a document's content and chunks for a new version."""
        with self._db.transaction() as conn:
            conn.execute("DELETE FROM document_chunks WHERE document_id = %s", [document_id])
            cur = conn.execute(
                """
                UPDATE documents
                SET filename = %s, file_hash = %s, file_size = %s, total_pages = %s,
                    total_chunks = %s, embedding_model = %s, chunk_size = %s,
                    chunk_overlap = %s, metadata = %s
                WHERE id = %s
                """,
                [
                    document.filename,
                    document.file_hash,
                    document.file_size,
                    document.total_pages,
                    len(chunks),
                    document.embedding_model,
                    document.chunk_size,
                    document.chunk_overlap,
                    Jsonb(document.metadata),
                    document_id,
                ],
            )
            if cur.rowcount != 1:
                raise DatabaseError(f"Document {document_id} no longer exists; cannot replace it")
            self._insert_chunks(conn, document_id, chunks)

    def rename_documents(self, renames: Sequence[tuple[UUID, str]]) -> None:
        """Update filenames in one transaction (safe even when names are swapped)."""
        if not renames:
            return
        with self._db.transaction() as conn:
            with conn.cursor() as cur:
                # Move every row to a unique placeholder first so a swap of two
                # names never trips the UNIQUE(filename) constraint midway.
                cur.executemany(
                    "UPDATE documents SET filename = '__renaming__/' || id::text WHERE id = %s",
                    [(doc_id,) for doc_id, _ in renames],
                )
                cur.executemany(
                    "UPDATE documents SET filename = %s WHERE id = %s",
                    [(name, doc_id) for doc_id, name in renames],
                )

    def delete_documents(self, document_ids: Sequence[UUID]) -> int:
        """Delete documents; their chunks go with them via ON DELETE CASCADE."""
        if not document_ids:
            return 0
        with self._db.transaction() as conn:
            cur = conn.execute("DELETE FROM documents WHERE id = ANY(%s)", [list(document_ids)])
            return cur.rowcount

    @staticmethod
    def _insert_chunks(conn: psycopg.Connection, document_id: UUID, chunks: Sequence[NewChunk]) -> None:
        if not chunks:
            return
        with conn.cursor() as cur:
            cur.executemany(
                _INSERT_CHUNK_SQL,
                [
                    (
                        document_id,
                        c.chunk_index,
                        c.content,
                        c.page_number,
                        c.page_end,
                        c.token_count,
                        c.embedding,
                    )
                    for c in chunks
                ],
            )

    # ------------------------------------------------------------------
    # Vector search
    # ------------------------------------------------------------------

    def search_similar(
        self,
        embedding: Sequence[float],
        *,
        limit: int,
        filters: SearchFilters | None = None,
        ef_search: int = 100,
    ) -> list[ChunkSearchResult]:
        """Nearest chunks by cosine distance using the HNSW index."""
        conditions: list[sql.Composable] = []
        params: dict[str, Any] = {"embedding": list(embedding), "limit": limit}
        if filters:
            if filters.filenames:
                conditions.append(sql.SQL("d.filename = ANY(%(filenames)s)"))
                params["filenames"] = list(filters.filenames)
            if filters.document_ids:
                conditions.append(sql.SQL("c.document_id = ANY(%(document_ids)s)"))
                params["document_ids"] = list(filters.document_ids)
            if filters.page_from is not None:
                conditions.append(sql.SQL("c.page_end >= %(page_from)s"))
                params["page_from"] = filters.page_from
            if filters.page_to is not None:
                conditions.append(sql.SQL("c.page_number <= %(page_to)s"))
                params["page_to"] = filters.page_to

        where = sql.SQL("WHERE ") + sql.SQL(" AND ").join(conditions) if conditions else sql.SQL("")
        query = sql.SQL(
            """
            SELECT c.id AS chunk_id, c.document_id, d.filename, c.chunk_index,
                   c.page_number, c.page_end, c.content, c.token_count,
                   1 - (c.embedding <=> %(embedding)s::vector) AS similarity
            FROM document_chunks AS c
            JOIN documents AS d ON d.id = c.document_id
            {where}
            ORDER BY c.embedding <=> %(embedding)s::vector
            LIMIT %(limit)s
            """
        ).format(where=where)

        with self._db.transaction() as conn:
            # ef_search must be >= LIMIT for HNSW to return LIMIT rows.
            conn.execute("SELECT set_config('hnsw.ef_search', %s, true)", [str(max(ef_search, limit))])
            if conditions and self._supports_iterative_scan(conn):
                # Keep scanning the index when filters discard candidates (pgvector >= 0.8).
                conn.execute("SELECT set_config('hnsw.iterative_scan', 'relaxed_order', true)")
            rows = conn.execute(query, params).fetchall()

        results = [
            ChunkSearchResult(
                chunk_id=r["chunk_id"],
                document_id=r["document_id"],
                filename=r["filename"],
                chunk_index=r["chunk_index"],
                page_number=r["page_number"],
                page_end=r["page_end"],
                content=r["content"],
                token_count=r["token_count"],
                similarity=float(r["similarity"]),
            )
            for r in rows
        ]
        # relaxed_order can return slightly out-of-order rows; restore strict order.
        results.sort(key=lambda r: r.similarity, reverse=True)
        return results

    def _supports_iterative_scan(self, conn: psycopg.Connection) -> bool:
        if self._iterative_scan_supported is None:
            row = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
            version = tuple(int(p) for p in row["extversion"].split(".")[:2]) if row else (0, 0)
            self._iterative_scan_supported = version >= (0, 8)
        return self._iterative_scan_supported

    # ------------------------------------------------------------------
    # Concurrency
    # ------------------------------------------------------------------

    @contextmanager
    def ingestion_lock(self) -> Iterator[None]:
        """Session-level advisory lock so two ingestion runs never overlap."""
        with self._db.connection() as conn:
            row = conn.execute("SELECT pg_try_advisory_lock(%s) AS locked", [_INGESTION_LOCK_ID]).fetchone()
            conn.commit()  # don't sit "idle in transaction" while holding the lock
            if not row or not row["locked"]:
                raise IngestionLockError("Another ingestion run is already in progress")
            try:
                yield
            finally:
                conn.execute("SELECT pg_advisory_unlock(%s)", [_INGESTION_LOCK_ID])
                conn.commit()
