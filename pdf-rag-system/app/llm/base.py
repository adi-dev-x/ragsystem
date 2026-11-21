"""Provider-neutral interface for answer generation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.core.retry import RetryPolicy, retry_call


@dataclass(frozen=True, slots=True)
class LLMResponse:
    text: str
    provider: str          # "claude" | "openai"
    model: str             # model id reported by the API
    input_tokens: int
    output_tokens: int
    stop_reason: str | None = None


class LLMProvider(ABC):
    """Every provider returns the same :class:`LLMResponse` shape."""

    name: str
    model: str

    def __init__(self, *, retry_policy: RetryPolicy | None = None) -> None:
        self._retry_policy = retry_policy or RetryPolicy()

    def generate(self, *, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Generate an answer, retrying transient failures with backoff."""
        return retry_call(
            lambda: self._generate_once(system_prompt=system_prompt, user_prompt=user_prompt),
            policy=self._retry_policy,
            operation=f"llm:{self.name}",
        )

    @abstractmethod
    def _generate_once(self, *, system_prompt: str, user_prompt: str) -> LLMResponse:
        """One API request. Raise ``LLMError(retryable=...)`` on failure."""
