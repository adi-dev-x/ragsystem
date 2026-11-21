"""Retry with exponential backoff and jitter for transient upstream failures."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable, TypeVar

from app.core.exceptions import ExternalServiceError
from app.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 5
    base_delay: float = 1.0
    max_delay: float = 30.0
    jitter: bool = True

    def delay_for(self, attempt: int, retry_after: float | None = None) -> float:
        """Delay before the next attempt; ``attempt`` is the 1-based attempt that failed."""
        backoff = min(self.max_delay, self.base_delay * (2 ** (attempt - 1)))
        if self.jitter and backoff > 0:
            # "Equal jitter": keep half the backoff, randomise the other half.
            backoff = backoff / 2 + random.uniform(0, backoff / 2)
        if retry_after is not None and retry_after > 0:
            backoff = max(backoff, retry_after)
        return min(backoff, self.max_delay)


def is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, ExternalServiceError) and exc.retryable


def retry_call(
    fn: Callable[[], T],
    *,
    policy: RetryPolicy,
    operation: str,
    should_retry: Callable[[BaseException], bool] = is_retryable,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call ``fn`` and retry transient failures according to ``policy``."""
    attempt = 1
    while True:
        try:
            return fn()
        except Exception as exc:
            if attempt >= policy.max_attempts or not should_retry(exc):
                raise
            delay = policy.delay_for(attempt, getattr(exc, "retry_after", None))
            logger.warning(
                "transient failure, retrying",
                extra={
                    "operation": operation,
                    "attempt": attempt,
                    "max_attempts": policy.max_attempts,
                    "delay_seconds": round(delay, 2),
                    "error": str(exc),
                },
            )
            sleep(delay)
            attempt += 1
