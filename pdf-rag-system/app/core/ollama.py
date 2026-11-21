"""Minimal HTTP client for a local Ollama server (https://github.com/ollama/ollama)."""

from __future__ import annotations

from typing import Any

import httpx

from app.core.exceptions import ExternalServiceError
from app.core.http_errors import is_retryable_status, retry_after_seconds


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout: float,
        retry_on_timeout: bool = True,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout
        # Chat generation on a slow machine takes as long on every attempt, so
        # retrying a timeout only multiplies the wait; embeddings are quick.
        self._retry_on_timeout = retry_on_timeout
        self._http = httpx.Client(base_url=self.base_url, timeout=timeout, transport=transport)

    def post(self, path: str, payload: dict[str, Any], *, error_cls: type[ExternalServiceError]) -> dict[str, Any]:
        """POST JSON and return the decoded body, mapping failures to ``error_cls``."""
        try:
            response = self._http.post(path, json=payload)
        except httpx.ConnectError as exc:
            # Ollama not running is a setup problem; retrying won't help.
            raise error_cls(
                f"Cannot reach Ollama at {self.base_url}. Is it running? Start it with `ollama serve` "
                "or open the Ollama app."
            ) from exc
        except httpx.TimeoutException as exc:
            if self._retry_on_timeout:
                raise error_cls("Ollama request timed out (the model may still be loading)", retryable=True) from exc
            raise error_cls(
                f"Ollama did not answer within {self._timeout:g}s. The model is too slow for this prompt on "
                "this machine: raise LLM_TIMEOUT_SECONDS, lower MAX_CONTEXT_TOKENS or TOP_K, or use a smaller "
                "or GPU-backed model."
            ) from exc
        except httpx.HTTPError as exc:
            raise error_cls(f"Ollama request failed: {exc.__class__.__name__}", retryable=True) from exc

        if response.status_code >= 400:
            message = _error_message(response)
            model = payload.get("model")
            if response.status_code == 404 and model:
                message = f"Ollama model {model!r} not found. Download it with `ollama pull {model}`."
            raise error_cls(
                f"Ollama returned HTTP {response.status_code}: {message}",
                retryable=is_retryable_status(response.status_code),
                retry_after=retry_after_seconds(response),
                details={"status_code": response.status_code},
            )
        try:
            return response.json()
        except ValueError as exc:
            raise error_cls("Ollama returned a response that is not JSON", retryable=True) from exc

    def close(self) -> None:
        self._http.close()


def _error_message(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:300] or response.reason_phrase
    if isinstance(body, dict) and body.get("error"):
        return str(body["error"])[:300]
    return response.reason_phrase
