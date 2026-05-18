from __future__ import annotations
import functools
import time
from typing import Callable, TypeVar

T = TypeVar("T")


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    exceptions: tuple = (Exception,),
) -> Callable:
    """Decorator: retry on specified exceptions with exponential back-off."""
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            delay = base_delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions:
                    if attempt == max_attempts:
                        raise
                    time.sleep(delay)
                    delay = min(delay * 2, max_delay)
        return wrapper
    return decorator
