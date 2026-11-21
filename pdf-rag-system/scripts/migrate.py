"""Apply or inspect database migrations.

    python -m scripts.migrate           # apply pending migrations
    python -m scripts.migrate --status  # list migrations and whether they are applied
    python -m scripts.migrate --set-embedding-dimensions 768 [--yes]
                                        # resize the vector column (deletes indexed documents)
"""

from __future__ import annotations

import argparse
import sys

from pydantic import ValidationError

from app.config import get_settings
from app.core.exceptions import RAGError
from app.core.logging import configure_logging, get_logger
from app.vectorstore.migrations import (
    apply_migrations,
    count_indexed_documents,
    migration_status,
    resize_embedding_dimension,
)

logger = get_logger("scripts.migrate")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Database migrations")
    parser.add_argument("--status", action="store_true", help="Show migration status only")
    parser.add_argument(
        "--set-embedding-dimensions",
        type=int,
        metavar="N",
        help="Change the embedding column to vector(N). Deletes indexed documents; re-ingest afterwards.",
    )
    parser.add_argument("--yes", action="store_true", help="Confirm deleting indexed documents")
    args = parser.parse_args(argv)

    try:
        settings = get_settings()
    except ValidationError as exc:
        print(f"Invalid configuration:\n{exc}", file=sys.stderr)
        return 2
    configure_logging(settings.log_level, settings.log_format)
    url = settings.database_url.get_secret_value()

    try:
        if args.set_embedding_dimensions is not None:
            apply_migrations(url)
            existing = count_indexed_documents(url)
            if existing and not args.yes:
                print(
                    f"{existing} indexed document(s) would be deleted, because embeddings of a different size "
                    "can't be compared. Re-run with --yes to continue, then run `python -m scripts.ingest`.",
                    file=sys.stderr,
                )
                return 1
            removed = resize_embedding_dimension(url, args.set_embedding_dimensions)
            print(
                f"Embedding column is now vector({args.set_embedding_dimensions}); removed {removed} document(s). "
                f"Set EMBEDDING_DIMENSIONS={args.set_embedding_dimensions} and run `python -m scripts.ingest`."
            )
        elif args.status:
            for item in migration_status(url):
                print(f"{'applied' if item.applied else 'pending':<8} {item.name}")
        else:
            applied = apply_migrations(url)
            print(f"Applied {len(applied)} migration(s)" if applied else "Schema is up to date")
    except RAGError as exc:
        logger.error("migration command failed", extra={"code": exc.code, "error": exc.message})
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
