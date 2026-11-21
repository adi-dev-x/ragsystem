"""Answer generation with Anthropic Claude (Messages API)."""

from __future__ import annotations

from typing import Any

import anthropic

from app.core.exceptions import ConfigurationError, LLMError, LLMRefusalError
from app.core.http_errors import is_retryable_status, retry_after_seconds
from app.core.logging import get_logger
from app.core.retry import RetryPolicy
from app.llm.base import LLMProvider, LLMResponse

logger = get_logger(__name__)

# Server-side refusal fallback: if a safety classifier declines, the API re-runs
# the request on Anthropic's recommended fallback model within the same call.
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class ClaudeProvider(LLMProvider):
    name = "claude"

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str | None,
        max_output_tokens: int,
        timeout: float,
        effort: str | None = None,
        refusal_fallback: bool = True,
        retry_policy: RetryPolicy | None = None,
        client: Any | None = None,
    ) -> None:
        super().__init__(retry_policy=retry_policy)
        if not model:
            raise ConfigurationError("ANTHROPIC_MODEL must be set when LLM_PROVIDER=claude")
        if client is None and not api_key:
            raise ConfigurationError("ANTHROPIC_API_KEY must be set when LLM_PROVIDER=claude")
        self.model = model
        self._max_output_tokens = max_output_tokens
        self._effort = effort
        self._refusal_fallback = refusal_fallback
        # SDK retries are disabled: LLMProvider.generate owns retry/backoff.
        self._client = client or anthropic.Anthropic(api_key=api_key, max_retries=0, timeout=timeout)

    def _request_kwargs(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self._max_output_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        if self._effort:
            kwargs["output_config"] = {"effort": self._effort}
        if self._refusal_fallback:
            kwargs["betas"] = [_FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        return kwargs

    def _generate_once(self, *, system_prompt: str, user_prompt: str) -> LLMResponse:
        kwargs = self._request_kwargs(system_prompt, user_prompt)
        messages_api = self._client.beta.messages if self._refusal_fallback else self._client.messages
        try:
            response = messages_api.create(**kwargs)
        except anthropic.RateLimitError as exc:
            raise LLMError(
                "Anthropic rate limit exceeded",
                retryable=True,
                retry_after=retry_after_seconds(exc.response),
            ) from exc
        except anthropic.APITimeoutError as exc:
            raise LLMError("Anthropic request timed out", retryable=True) from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError("Could not reach the Anthropic API", retryable=True) from exc
        except anthropic.AuthenticationError as exc:
            raise LLMError("Anthropic rejected the API key (authentication failed)") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(
                f"Anthropic request failed with HTTP {exc.status_code}",
                retryable=is_retryable_status(exc.status_code),
                retry_after=retry_after_seconds(exc.response),
                details={"status_code": exc.status_code},
            ) from exc

        # Check the stop reason before reading content.
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise LLMRefusalError(
                "The model declined to answer this request",
                details={"category": category} if category else None,
            )

        # Content may include thinking blocks; only text blocks form the answer.
        text = "".join(block.text for block in response.content if block.type == "text").strip()
        if not text:
            raise LLMError("Anthropic returned an empty answer", retryable=True)
        if response.stop_reason == "max_tokens":
            logger.warning("answer truncated at max output tokens", extra={"provider": self.name})

        return LLMResponse(
            text=text,
            provider=self.name,
            model=response.model or self.model,
            input_tokens=int(response.usage.input_tokens or 0),
            output_tokens=int(response.usage.output_tokens or 0),
            stop_reason=response.stop_reason,
        )
