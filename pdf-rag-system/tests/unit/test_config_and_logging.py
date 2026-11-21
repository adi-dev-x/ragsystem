import json
import logging

import pytest
from pydantic import ValidationError

from app.core.logging import JsonFormatter, RedactionFilter, redact


def test_defaults_match_specification(make_settings) -> None:
    s = make_settings()
    assert (s.chunk_size, s.chunk_overlap, s.top_k) == (800, 150, 5)
    assert s.embedding_model == "text-embedding-3-small" and s.embedding_dimensions == 1536
    assert s.pdf_directory_path.name == "dataloads"


def test_overlap_must_be_smaller_than_chunk_size(make_settings) -> None:
    with pytest.raises(ValidationError):
        make_settings(chunk_size=100, chunk_overlap=100)


def test_sqlalchemy_style_url_is_normalised(make_settings) -> None:
    s = make_settings(database_url="postgresql+psycopg://u:p@localhost/db")
    assert s.database_url.get_secret_value() == "postgresql://u:p@localhost/db"


def test_secrets_are_not_shown_in_repr(make_settings) -> None:
    s = make_settings(openai_api_key="sk-proj-supersecretvalue1234567890")
    assert "supersecret" not in repr(s)
    assert "secret@" not in repr(s)


@pytest.mark.parametrize(
    "raw",
    [
        "key sk-proj-abcdefghijklmnopqrstuvwxyz123",
        "key sk-ant-api03-abcdefghijklmnop",
        "postgresql://rag:hunter2@localhost:5432/rag",
        "Authorization: Bearer abcdefghijklmnop",
        "api_key=abcd1234efgh",
    ],
)
def test_redaction(raw: str) -> None:
    cleaned = redact(raw)
    for secret in ("abcdefghijklmnopqrstuvwxyz123", "api03-abcdefghijklmnop", "hunter2", "abcdefghijklmnop", "abcd1234efgh"):
        assert secret not in cleaned


def test_json_formatter_includes_extras_and_redacts() -> None:
    record = logging.LogRecord("t", logging.INFO, __file__, 1, "connecting to %s", ("postgresql://u:pw123@h/db",), None)
    record.document = "a.pdf"
    record.api = "sk-ant-api03-secretsecretsecret"
    RedactionFilter().filter(record)
    payload = json.loads(JsonFormatter().format(record))
    assert payload["message"] == "connecting to postgresql://u:***@h/db"
    assert payload["document"] == "a.pdf"
    assert "secretsecret" not in payload["api"]
