"""Ingest PDFs from the configured directory into PostgreSQL.

    python -m scripts.ingest              # sync dataloads/ into the database
    python -m scripts.ingest --dry-run    # show what would change, no writes/API calls
    python -m scripts.ingest --force      # re-chunk and re-embed every PDF
    python -m scripts.ingest --prune      # also delete indexed documents whose PDF was removed

By default a PDF can be deleted from the folder (to free space) and its indexed
content stays searchable.

Exit codes: 0 success, 1 some files failed, 2 fatal error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from app.config import Settings, get_settings
from app.core.exceptions import ConfigurationError, RAGError
from app.core.logging import configure_logging, get_logger
from app.core.tokenizer import TiktokenTokenizer
from app.embeddings.embedding_service import create_embedding_service
from app.ingestion.chunker import TextChunker
from app.ingestion.pipeline import IngestionPipeline, IngestionReport
from app.vectorstore.database import Database
from app.vectorstore.migrations import apply_migrations
from app.vectorstore.repository import DocumentRepository

logger = get_logger("scripts.ingest")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest PDFs into the RAG vector store.")
    parser.add_argument("--directory", type=Path, help="PDF directory (overrides PDF_DIRECTORY)")
    parser.add_argument("--force", action="store_true", help="Re-process every PDF even if unchanged")
    parser.add_argument("--dry-run", action="store_true", help="Report planned changes without writing")
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Also delete indexed documents whose PDF is no longer in the folder (default: keep them)",
    )
    parser.add_argument("--skip-migrations", action="store_true", help="Don't apply pending migrations first")
    return parser.parse_args(argv)


def verify_embedding_dimension(repository: DocumentRepository, settings: Settings) -> None:
    db_dims = repository.get_embedding_dimension()
    if db_dims is None:
        raise ConfigurationError("document_chunks table not found; run `python -m scripts.migrate`")
    if db_dims != settings.embedding_dimensions:
        raise ConfigurationError(
            f"EMBEDDING_DIMENSIONS={settings.embedding_dimensions} but document_chunks.embedding is "
            f"vector({db_dims}). Run `python -m scripts.migrate --set-embedding-dimensions "
            f"{settings.embedding_dimensions}` and re-ingest."
        )


def run(settings: Settings, args: argparse.Namespace) -> IngestionReport:
    database_url = settings.database_url.get_secret_value()
    # A dry run performs no writes at all, schema changes included.
    if not args.skip_migrations and not args.dry_run:
        apply_migrations(database_url)

    directory = args.directory.resolve() if args.directory else settings.pdf_directory_path
    with Database(settings, application_name="pdf-rag-ingest") as database:
        repository = DocumentRepository(database)
        verify_embedding_dimension(repository, settings)
        pipeline = IngestionPipeline(
            repository=repository,
            embedding_service=None if args.dry_run else create_embedding_service(settings),
            chunker=TextChunker(
                TiktokenTokenizer(settings.tokenizer_encoding),
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            ),
            pdf_directory=directory,
            embedding_model=settings.embedding_model,
            max_pdf_size_bytes=settings.max_pdf_size_bytes,
        )
        return pipeline.run(force=args.force, dry_run=args.dry_run, prune=args.prune)


_PLAN_LABELS = {
    "unchanged": "Unchanged",
    "new": "New",
    "modified": "Modified (re-embed)",
    "config_changed": "Settings changed (re-embed)",
    "forced": "Forced (re-embed)",
    "renamed": "Renamed",
    "deleted": "Deleted (--prune)",
    "kept_missing": "PDF removed, index kept",
    "duplicates": "Duplicate content",
    "kept_invalid": "Unreadable (index kept)",
}


def _print_summary(report: IngestionReport) -> None:
    if report.dry_run:
        title = "Ingestion dry run (no changes made)"
        print(f"\n{title}\n" + "-" * len(title), file=sys.stderr)
        print(f"  {'PDFs discovered':<28}{report.discovered}", file=sys.stderr)
        for key, value in report.planned.items():
            print(f"  {_PLAN_LABELS.get(key, key):<28}{value}", file=sys.stderr)
        for filename, error in report.errors.items():
            print(f"  ! {filename}: {error}", file=sys.stderr)
        print(json.dumps(report.to_dict()))
        return

    title = "Ingestion summary"
    rows = [
        ("PDFs discovered", report.discovered),
        ("Unchanged (skipped)", report.unchanged),
        ("New documents", report.inserted),
        ("Updated documents", report.updated),
        ("Renamed", report.renamed),
        ("Deleted", report.deleted),
        ("Duplicate content", report.duplicates),
        ("Failed", report.failed),
        ("Chunks written", report.chunks_written),
        ("Embedding requests", report.embedding_requests),
        ("Embedding tokens", report.embedding_tokens),
        ("Duration (s)", report.duration_seconds),
    ]
    print(f"\n{title}\n" + "-" * len(title), file=sys.stderr)
    for label, value in rows:
        print(f"  {label:<22}{value}", file=sys.stderr)
    for filename, error in report.errors.items():
        print(f"  ! {filename}: {error}", file=sys.stderr)
    # Machine-readable result on stdout.
    print(json.dumps(report.to_dict()))


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        settings = get_settings()
    except ValidationError as exc:
        print(f"Invalid configuration:\n{exc}", file=sys.stderr)
        return 2
    configure_logging(settings.log_level, settings.log_format)

    try:
        report = run(settings, args)
    except RAGError as exc:
        logger.error("ingestion aborted", extra={"code": exc.code, "error": exc.message})
        return 2
    except KeyboardInterrupt:
        logger.warning("ingestion interrupted")
        return 2

    _print_summary(report)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
