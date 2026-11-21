import pytest

from app.core.exceptions import ExternalServiceError
from app.core.retry import RetryPolicy, retry_call


def test_backoff_grows_exponentially_and_is_capped() -> None:
    policy = RetryPolicy(max_attempts=10, base_delay=1.0, max_delay=8.0, jitter=False)
    assert [policy.delay_for(a) for a in range(1, 6)] == [1.0, 2.0, 4.0, 8.0, 8.0]


def test_jitter_stays_within_bounds() -> None:
    policy = RetryPolicy(base_delay=2.0, max_delay=100.0, jitter=True)
    for _ in range(100):
        assert 2.0 <= policy.delay_for(2) <= 4.0


def test_retry_after_header_is_honoured() -> None:
    policy = RetryPolicy(base_delay=0.5, max_delay=30.0, jitter=False)
    assert policy.delay_for(1, retry_after=7.0) == 7.0
    assert policy.delay_for(1, retry_after=90.0) == 30.0


def test_retry_call_retries_only_retryable_errors() -> None:
    sleeps: list[float] = []
    attempts = {"n": 0}

    def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ExternalServiceError("transient", retryable=True)
        return "ok"

    policy = RetryPolicy(max_attempts=5, base_delay=1.0, max_delay=10.0, jitter=False)
    assert retry_call(flaky, policy=policy, operation="test", sleep=sleeps.append) == "ok"
    assert sleeps == [1.0, 2.0]

    def permanent() -> None:
        raise ExternalServiceError("bad request", retryable=False)

    with pytest.raises(ExternalServiceError):
        retry_call(permanent, policy=policy, operation="test", sleep=sleeps.append)
    assert sleeps == [1.0, 2.0]  # no extra sleeps


def test_retry_call_reraises_after_last_attempt() -> None:
    calls = []

    def always_fails() -> None:
        calls.append(1)
        raise ExternalServiceError("down", retryable=True)

    with pytest.raises(ExternalServiceError):
        retry_call(always_fails, policy=RetryPolicy(max_attempts=3, base_delay=0, jitter=False), operation="t", sleep=lambda _: None)
    assert len(calls) == 3
