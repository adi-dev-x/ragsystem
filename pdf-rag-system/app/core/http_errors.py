"""Helpers for classifying upstream HTTP failures as transient or permanent."""

from __future__ import annotations

from typing import Any

# 408 timeout, 409 conflict/lock, 429 rate limit, 5xx server errors, 529 overloaded.
RETRYABLE_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504, 529})


def is_retryable_status(status_code: int | None) -> bool:
    return status_code is not None and (status_code in RETRYABLE_STATUS_CODES or status_code >= 500)


def retry_after_seconds(response: Any) -> float | None:
    """Parse ``retry-after-ms`` / ``retry-after`` headers (seconds form only)."""
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    for name, scale in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
        value = headers.get(name)
        if value is None:
            continue
        try:
            seconds = float(value) * scale
        except (TypeError, ValueError):
            continue
        if seconds >= 0:
            return seconds
    return None
