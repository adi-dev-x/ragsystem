"""Answer generation with a local Ollama chat model."""

from __future__ import annotations

import re
from typing import Any

from app.core.exceptions import ConfigurationError, LLMError
from app.core.logging import get_logger
from app.core.ollama import OllamaClient
from app.core.retry import RetryPolicy
from app.llm.base import LLMProvider, LLMResponse

logger = get_logger(__name__)

# Reasoning models (qwen3, deepseek-r1, ...) may put their reasoning in <think> tags.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(
        self,
        *,
        client: OllamaClient,
        model: str | None,
        max_output_tokens: int,
        num_ctx: int,
        temperature: float | None = None,
        keep_alive: str | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        super().__init__(retry_policy=retry_policy)
        if not model:
            raise ConfigurationError("OLLAMA_MODEL must be set when LLM_PROVIDER=ollama")
        self.model = model
        self._client = client
        self._max_output_tokens = max_output_tokens
        self._num_ctx = num_ctx
        self._temperature = temperature
        self._keep_alive = keep_alive

    def _generate_once(self, *, system_prompt: str, user_prompt: str) -> LLMResponse:
        options: dict[str, Any] = {"num_ctx": self._num_ctx, "num_predict": self._max_output_tokens}
        if self._temperature is not None:
            options["temperature"] = self._temperature
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": options,
        }
        if self._keep_alive:
            payload["keep_alive"] = self._keep_alive

        data = self._client.post("/api/chat", payload, error_cls=LLMError)

        message = data.get("message") or {}
        text = _THINK_RE.sub("", str(message.get("content") or "")).strip()
        if not text:
            raise LLMError("Ollama returned an empty answer", retryable=True)

        input_tokens = int(data.get("prompt_eval_count") or 0)
        if input_tokens >= self._num_ctx:
            logger.warning(
                "prompt filled the Ollama context window and may have been truncated; raise OLLAMA_NUM_CTX "
                "or lower MAX_CONTEXT_TOKENS",
                extra={"prompt_tokens": input_tokens, "num_ctx": self._num_ctx},
            )
        if data.get("done_reason") == "length":
            logger.warning("answer truncated at max output tokens", extra={"provider": self.name})

        return LLMResponse(
            text=text,
            provider=self.name,
            model=str(data.get("model") or self.model),
            input_tokens=input_tokens,
            output_tokens=int(data.get("eval_count") or 0),
            stop_reason=data.get("done_reason"),
        )
