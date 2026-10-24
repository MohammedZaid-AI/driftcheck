"""Retry with exponential backoff for rate limits and timeouts."""

from __future__ import annotations

import random
import time
from typing import Callable, TypeVar

from driftcheck.log import get_logger

T = TypeVar("T")
log = get_logger(__name__)

MAX_TRIES = 4
RETRYABLE_CODES = {408, 429, 500, 502, 503, 504}


def is_retryable(error: BaseException) -> bool:
    """True for rate limits, timeouts and temporary server errors."""
    if isinstance(error, (TimeoutError, ConnectionError)):
        return True
    # google-genai APIError has .code; httpx errors carry a response.
    code = getattr(error, "code", None) or getattr(error, "status_code", None)
    if isinstance(code, int) and code in RETRYABLE_CODES:
        return True
    return "Timeout" in type(error).__name__


def retry_call(
    fn: Callable[[], T],
    *,
    max_tries: int = MAX_TRIES,
    base_delay: float = 2.0,
    max_delay: float = 30.0,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call fn, retrying retryable errors up to max_tries in total."""
    for attempt in range(1, max_tries + 1):
        try:
            return fn()
        except Exception as error:
            if attempt == max_tries or not is_retryable(error):
                raise
            delay = min(max_delay, base_delay * 2 ** (attempt - 1))
            delay *= random.uniform(0.8, 1.2)
            log.warning("Try %d of %d failed with %s. Retrying in %.1fs.", attempt, max_tries, error, delay)
            sleep(delay)
    raise AssertionError("unreachable")
