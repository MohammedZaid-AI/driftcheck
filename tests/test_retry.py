from __future__ import annotations

import pytest

from driftcheck.retry import is_retryable, retry_call


class FakeAPIError(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(f"code {code}")
        self.code = code


def _flaky(errors: list[Exception], result: str = "ok"):
    calls = {"n": 0}

    def fn() -> str:
        calls["n"] += 1
        if errors:
            raise errors.pop(0)
        return result

    return fn, calls


def test_retries_rate_limit_then_succeeds() -> None:
    sleeps: list[float] = []
    fn, calls = _flaky([FakeAPIError(429), TimeoutError()])
    assert retry_call(fn, sleep=sleeps.append) == "ok"
    assert calls["n"] == 3
    assert len(sleeps) == 2
    assert sleeps[1] > sleeps[0]


def test_gives_up_after_four_tries() -> None:
    fn, calls = _flaky([FakeAPIError(503)] * 10)
    with pytest.raises(FakeAPIError):
        retry_call(fn, sleep=lambda _: None)
    assert calls["n"] == 4


def test_does_not_retry_bad_request() -> None:
    fn, calls = _flaky([FakeAPIError(400)])
    with pytest.raises(FakeAPIError):
        retry_call(fn, sleep=lambda _: None)
    assert calls["n"] == 1


def test_is_retryable() -> None:
    assert is_retryable(FakeAPIError(429))
    assert is_retryable(TimeoutError())
    assert not is_retryable(FakeAPIError(401))
    assert not is_retryable(ValueError("bad"))
