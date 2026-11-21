-- 001_initial_schema.sql
-- Documents and their embedded chunks, stored with pgvector.

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------------
-- documents: one row per indexed PDF
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename        TEXT        NOT NULL,            -- path relative to PDF_DIRECTORY
    file_hash       TEXT        NOT NULL,            -- SHA-256 of the file bytes (hex)
    file_size       BIGINT      NOT NULL DEFAULT 0,
    total_pages     INTEGER     NOT NULL DEFAULT 0,
    total_chunks    INTEGER     NOT NULL DEFAULT 0,
    embedding_model TEXT        NOT NULL,
    chunk_size      INTEGER     NOT NULL,
    chunk_overlap   INTEGER     NOT NULL,
    metadata        JSONB       NOT NULL DEFAULT '{}'::jsonb,  -- PDF title, author, ...
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT documents_file_hash_key UNIQUE (file_hash),
    CONSTRAINT documents_filename_key  UNIQUE (filename),
    CONSTRAINT documents_file_hash_format CHECK (file_hash ~ '^[0-9a-f]{64}$'),
    CONSTRAINT documents_total_pages_nonneg  CHECK (total_pages >= 0),
    CONSTRAINT documents_total_chunks_nonneg CHECK (total_chunks >= 0)
);

-- ---------------------------------------------------------------------------
-- document_chunks: text chunks with their embeddings
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS document_chunks (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id  UUID        NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    chunk_index  INTEGER     NOT NULL,
    content      TEXT        NOT NULL,
    page_number  INTEGER     NOT NULL,   -- first page the chunk covers (1-based)
    page_end     INTEGER     NOT NULL,   -- last page the chunk covers
    token_count  INTEGER     NOT NULL,
    embedding    VECTOR(1536) NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT document_chunks_document_chunk_key UNIQUE (document_id, chunk_index),
    CONSTRAINT document_chunks_chunk_index_nonneg CHECK (chunk_index >= 0),
    CONSTRAINT document_chunks_page_range CHECK (page_number >= 1 AND page_end >= page_number),
    CONSTRAINT document_chunks_content_not_blank CHECK (length(btrim(content)) > 0)
);

-- Approximate nearest-neighbour search with cosine distance (<=>).
CREATE INDEX IF NOT EXISTS document_chunks_embedding_hnsw_idx
    ON document_chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Page-range filtering within a document.
CREATE INDEX IF NOT EXISTS document_chunks_document_page_idx
    ON document_chunks (document_id, page_number);

-- Listing documents newest-first.
CREATE INDEX IF NOT EXISTS documents_updated_at_idx
    ON documents (updated_at DESC);

-- ---------------------------------------------------------------------------
-- Keep documents.updated_at current on every UPDATE.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION set_updated_at() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at := now();
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS documents_set_updated_at ON documents;
CREATE TRIGGER documents_set_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();
