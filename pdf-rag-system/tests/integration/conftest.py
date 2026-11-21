"""Integration fixtures. These tests run only when TEST_DATABASE_URL points at a
PostgreSQL database with pgvector available. The database is wiped (public
schema dropped) before the session, so never point it at real data."""

from __future__ import annotations

import os
from collections.abc import Iterator

import psycopg
import pytest

from app.config import Settings
from app.vectorstore.database import Database
from app.vectorstore.migrations import apply_migrations
from app.vectorstore.repository import DocumentRepository

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

pytestmark = pytest.mark.integration


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if TEST_DATABASE_URL:
        return
    skip = pytest.mark.skip(reason="set TEST_DATABASE_URL to run PostgreSQL integration tests")
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(skip)


@pytest.fixture(scope="session")
def database_url() -> str:
    assert TEST_DATABASE_URL
    with psycopg.connect(TEST_DATABASE_URL, autocommit=True) as conn:
        conn.execute("DROP SCHEMA IF EXISTS public CASCADE")
        conn.execute("CREATE SCHEMA public")
    apply_migrations(TEST_DATABASE_URL)
    return TEST_DATABASE_URL


@pytest.fixture(scope="session")
def pg_settings(database_url: str) -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        database_url=database_url,
        db_pool_min_size=1,
        db_pool_max_size=4,
        anthropic_model="test",
        anthropic_api_key="test",
        openai_api_key="test",
    )


@pytest.fixture(scope="session")
def database(pg_settings: Settings) -> Iterator[Database]:
    db = Database(pg_settings, application_name="pdf-rag-tests")
    db.open(wait=True)
    yield db
    db.close()


@pytest.fixture
def repository(database: Database) -> Iterator[DocumentRepository]:
    with database.connection() as conn:
        conn.execute("TRUNCATE documents CASCADE")
    yield DocumentRepository(database)
