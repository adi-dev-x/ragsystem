"""Select the answer-generation provider from ``LLM_PROVIDER``."""

from __future__ import annotations

from pydantic import SecretStr

from app.config import Settings
from app.core.exceptions import ConfigurationError
from app.core.logging import get_logger
from app.core.retry import RetryPolicy
from app.llm.base import LLMProvider

logger = get_logger(__name__)


def _secret(value: SecretStr | None) -> str | None:
    return value.get_secret_value() if value else None


def create_llm_provider(settings: Settings) -> LLMProvider:
    retry_policy = RetryPolicy(
        max_attempts=settings.retry_max_attempts,
        base_delay=settings.retry_base_delay_seconds,
        max_delay=settings.retry_max_delay_seconds,
    )
    if settings.llm_provider == "claude":
        from app.llm.claude_provider import ClaudeProvider

        return ClaudeProvider(
            api_key=_secret(settings.anthropic_api_key),
            model=settings.anthropic_model,
            max_output_tokens=settings.llm_max_output_tokens,
            timeout=settings.llm_timeout_seconds,
            effort=settings.anthropic_effort,
            refusal_fallback=settings.anthropic_refusal_fallback,
            retry_policy=retry_policy,
        )
    if settings.llm_provider == "openai":
        from app.llm.openai_provider import OpenAIProvider

        return OpenAIProvider(
            api_key=_secret(settings.openai_api_key),
            model=settings.openai_model,
            max_output_tokens=settings.llm_max_output_tokens,
            timeout=settings.llm_timeout_seconds,
            temperature=settings.openai_temperature,
            retry_policy=retry_policy,
        )
    if settings.llm_provider == "ollama":
        from app.core.ollama import OllamaClient
        from app.llm.ollama_provider import OllamaProvider

        # Leave room for the system prompt, question and answer inside num_ctx.
        if settings.max_context_tokens + 1500 > settings.ollama_num_ctx:
            logger.warning(
                "MAX_CONTEXT_TOKENS leaves little room in OLLAMA_NUM_CTX; long prompts may be truncated",
                extra={"max_context_tokens": settings.max_context_tokens, "ollama_num_ctx": settings.ollama_num_ctx},
            )
        return OllamaProvider(
            client=OllamaClient(settings.ollama_base_url, timeout=settings.llm_timeout_seconds, retry_on_timeout=False),
            model=settings.ollama_model,
            max_output_tokens=settings.llm_max_output_tokens,
            num_ctx=settings.ollama_num_ctx,
            temperature=settings.ollama_temperature,
            keep_alive=settings.ollama_keep_alive,
            retry_policy=retry_policy,
        )
    raise ConfigurationError(
        f"Unsupported LLM_PROVIDER: {settings.llm_provider!r} (use 'claude', 'openai' or 'ollama')"
    )
