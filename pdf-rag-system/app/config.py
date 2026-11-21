"""Application configuration loaded from environment variables and `.env`."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EffortLevel = Literal["low", "medium", "high", "xhigh", "max"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # Treat `SIMILARITY_THRESHOLD=` as "not set" so the default applies.
        env_ignore_empty=True,
        case_sensitive=False,
    )

    # --- Database -----------------------------------------------------------
    database_url: SecretStr
    db_pool_min_size: int = Field(default=1, ge=0)
    db_pool_max_size: int = Field(default=10, ge=1)
    db_pool_timeout_seconds: float = Field(default=10.0, gt=0)
    db_statement_timeout_ms: int = Field(default=30_000, ge=0)

    # --- API keys -----------------------------------------------------------
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None

    # --- Embeddings ---------------------------------------------------------
    embedding_provider: Literal["openai", "ollama"] = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = Field(default=1536, gt=0, le=2000)
    embedding_batch_size: int = Field(default=100, ge=1, le=2048)
    embedding_timeout_seconds: float = Field(default=60.0, gt=0)
    # Some models are trained with task prefixes, e.g. nomic-embed-text uses
    # "search_document: " for passages and "search_query: " for questions.
    embedding_document_prefix: str = ""
    embedding_query_prefix: str = ""

    # --- Answer generation --------------------------------------------------
    llm_provider: Literal["openai", "claude", "ollama"] = "claude"
    openai_model: str | None = None
    anthropic_model: str | None = None
    ollama_model: str | None = None
    llm_max_output_tokens: int = Field(default=16_000, ge=1)
    llm_timeout_seconds: float = Field(default=120.0, gt=0)
    openai_temperature: float | None = Field(default=None, ge=0, le=2)
    anthropic_effort: EffortLevel | None = None
    anthropic_refusal_fallback: bool = True
    # How the answer text is written:
    #   plain    - plain sentences and "- " lists, no Markdown symbols (shown to users as is)
    #   markdown - headings, bold and bullets, for clients that render Markdown
    #   spoken   - natural conversational speech for text-to-speech: no citations,
    #              headings, lists or symbols (citation_style is ignored)
    answer_format: Literal["plain", "markdown", "spoken"] = "plain"
    # How the answer cites sources (plain and markdown formats only):
    #   numbered - short [1], [2] markers matching "id" in the response's sources
    #   document - [file.pdf, p. 12] written out in the text
    citation_style: Literal["numbered", "document"] = "numbered"

    # --- Ollama (local models, used when a provider above is "ollama") -------
    ollama_base_url: str = "http://localhost:11434"
    # Context window for answer generation. Ollama's own default is small and
    # silently truncates long prompts, so it is always sent explicitly.
    ollama_num_ctx: int = Field(default=8192, ge=512)
    ollama_temperature: float | None = Field(default=None, ge=0, le=2)
    ollama_keep_alive: str | None = None  # e.g. "10m"; how long Ollama keeps the model loaded

    # --- Chunking -----------------------------------------------------------
    chunk_size: int = Field(default=800, ge=50, le=8000)
    chunk_overlap: int = Field(default=150, ge=0)
    tokenizer_encoding: str = "cl100k_base"

    # --- Retrieval ----------------------------------------------------------
    top_k: int = Field(default=5, ge=1, le=50)
    similarity_threshold: float = Field(default=0.3, ge=0, le=1)
    max_context_tokens: int = Field(default=6000, ge=100)
    hnsw_ef_search: int = Field(default=100, ge=1, le=1000)
    retrieval_candidate_multiplier: int = Field(default=3, ge=1, le=10)

    # --- Ingestion ----------------------------------------------------------
    pdf_directory: Path = Path("dataloads")
    max_pdf_size_mb: float = Field(default=200, gt=0)

    # --- Retries ------------------------------------------------------------
    retry_max_attempts: int = Field(default=5, ge=1, le=20)
    retry_base_delay_seconds: float = Field(default=1.0, ge=0)
    retry_max_delay_seconds: float = Field(default=30.0, ge=0)

    # --- Logging ------------------------------------------------------------
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: Literal["json", "text"] = "json"

    @field_validator("llm_provider", mode="before")
    @classmethod
    def _normalise_provider(cls, value: object) -> object:
        if isinstance(value, str):
            value = value.strip().lower()
            if value == "anthropic":
                return "claude"
        return value

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper_log_level(cls, value: object) -> object:
        return value.strip().upper() if isinstance(value, str) else value

    @field_validator("database_url", mode="before")
    @classmethod
    def _normalise_database_url(cls, value: object) -> object:
        # Accept SQLAlchemy-style URLs such as postgresql+psycopg://...
        raw = value.get_secret_value() if isinstance(value, SecretStr) else value
        if isinstance(raw, str):
            for prefix in ("postgresql+psycopg://", "postgresql+psycopg2://", "postgres://"):
                if raw.startswith(prefix):
                    return "postgresql://" + raw[len(prefix):]
        return value

    @model_validator(mode="after")
    def _check_consistency(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if self.db_pool_min_size > self.db_pool_max_size:
            raise ValueError("DB_POOL_MIN_SIZE cannot exceed DB_POOL_MAX_SIZE")
        if self.retry_max_delay_seconds < self.retry_base_delay_seconds:
            raise ValueError("RETRY_MAX_DELAY_SECONDS must be >= RETRY_BASE_DELAY_SECONDS")
        return self

    @property
    def pdf_directory_path(self) -> Path:
        """PDF directory resolved against the project root when relative."""
        path = self.pdf_directory.expanduser()
        return path if path.is_absolute() else (PROJECT_ROOT / path)

    @property
    def max_pdf_size_bytes(self) -> int:
        return int(self.max_pdf_size_mb * 1024 * 1024)

    @property
    def llm_model(self) -> str | None:
        return {
            "claude": self.anthropic_model,
            "openai": self.openai_model,
            "ollama": self.ollama_model,
        }[self.llm_provider]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
