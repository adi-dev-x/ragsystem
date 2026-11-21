"""Answer generation with OpenAI chat models."""

from __future__ import annotations

from typing import Any

import openai

from app.core.exceptions import ConfigurationError, LLMError, LLMRefusalError
from app.core.http_errors import is_retryable_status, retry_after_seconds
from app.core.logging import get_logger
from app.core.retry import RetryPolicy
from app.llm.base import LLMProvider, LLMResponse

logger = get_logger(__name__)


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str | None,
        max_output_tokens: int,
        timeout: float,
        temperature: float | None = None,
        retry_policy: RetryPolicy | None = None,
        client: Any | None = None,
    ) -> None:
        super().__init__(retry_policy=retry_policy)
        if not model:
            raise ConfigurationError("OPENAI_MODEL must be set when LLM_PROVIDER=openai")
        if client is None and not api_key:
            raise ConfigurationError("OPENAI_API_KEY must be set when LLM_PROVIDER=openai")
        self.model = model
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._client = client or openai.OpenAI(api_key=api_key, max_retries=0, timeout=timeout)

    def _generate_once(self, *, system_prompt: str, user_prompt: str) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_completion_tokens": self._max_output_tokens,
        }
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature

        try:
            response = self._client.chat.completions.create(**kwargs)
        except openai.RateLimitError as exc:
            quota = getattr(exc, "code", None) == "insufficient_quota"
            raise LLMError(
                "OpenAI quota exhausted" if quota else "OpenAI rate limit exceeded",
                retryable=not quota,
                retry_after=retry_after_seconds(getattr(exc, "response", None)),
            ) from exc
        except openai.APITimeoutError as exc:
            raise LLMError("OpenAI request timed out", retryable=True) from exc
        except openai.APIConnectionError as exc:
            raise LLMError("Could not reach the OpenAI API", retryable=True) from exc
        except openai.AuthenticationError as exc:
            raise LLMError("OpenAI rejected the API key (authentication failed)") from exc
        except openai.APIStatusError as exc:
            raise LLMError(
                f"OpenAI request failed with HTTP {exc.status_code}",
                retryable=is_retryable_status(exc.status_code),
                retry_after=retry_after_seconds(exc.response),
                details={"status_code": exc.status_code},
            ) from exc

        if not response.choices:
            raise LLMError("OpenAI returned no choices", retryable=True)
        choice = response.choices[0]
        message = choice.message
        if getattr(message, "refusal", None):
            raise LLMRefusalError("The model declined to answer this request")
        text = (message.content or "").strip()
        if not text:
            raise LLMError("OpenAI returned an empty answer", retryable=True)
        if choice.finish_reason == "length":
            logger.warning("answer truncated at max output tokens", extra={"provider": self.name})

        usage = response.usage
        return LLMResponse(
            text=text,
            provider=self.name,
            model=response.model or self.model,
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            stop_reason=choice.finish_reason,
        )
