"""Cross-cutting helpers: logging, retry with backoff, rate limiting, timing."""
from __future__ import annotations

import functools
import logging
import logging.handlers
import random
import sys
import threading
import time
from collections import deque
from typing import Callable, Iterable, Iterator, Sequence, TypeVar

from .config import settings

T = TypeVar("T")

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s"
_configured = False


def setup_logging(level: str | None = None, to_file: bool = True) -> None:
    """Idempotent root logger setup. Console always, rotating file optionally."""
    global _configured
    if _configured:
        return

    root = logging.getLogger()
    root.setLevel(getattr(logging, (level or settings.log_level), logging.INFO))
    root.handlers.clear()

    # A cp1252 Windows console cannot encode emoji or box-drawing characters; a
    # single one in a log message would otherwise raise UnicodeEncodeError from
    # inside the logging call itself, which is a spectacularly confusing failure.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S"))
    root.addHandler(console)

    if to_file:
        try:
            settings.paths.ensure()
            fh = logging.handlers.RotatingFileHandler(
                settings.paths.logs / "screener.log",
                maxBytes=5_000_000,
                backupCount=3,
                encoding="utf-8",
            )
            fh.setFormatter(logging.Formatter(_LOG_FORMAT))
            root.addHandler(fh)
        except OSError:  # read-only fs, frozen exe, etc. Console is enough.
            pass

    # Third-party libraries are chatty; we only want their warnings.
    for noisy in ("urllib3", "requests", "websocket", "matplotlib", "numexpr"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)


log = get_logger(__name__)


# --------------------------------------------------------------------------
# Retry
# --------------------------------------------------------------------------
class TransientError(Exception):
    """Worth retrying: timeout, connection reset, 429, 5xx."""


class PermanentError(Exception):
    """Never worth retrying: bad credentials, malformed request, unknown symbol."""


def retry(
    attempts: int = 4,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    exceptions: tuple[type[BaseException], ...] = (TransientError, TimeoutError, ConnectionError),
    jitter: bool = True,
):
    """
    Exponential backoff with jitter.

    Two rules that matter more than the code:
      * PermanentError is re-raised immediately - retrying a 401 just burns time.
      * Jitter is not optional. Without it, N workers that fail together wake up
        together and hammer the recovering service in lockstep (thundering herd).
    """

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> T:
            last: BaseException | None = None
            for attempt in range(1, attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except PermanentError:
                    raise
                except exceptions as exc:
                    last = exc
                    if attempt == attempts:
                        break
                    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                    if jitter:
                        delay *= random.uniform(0.5, 1.5)
                    log.warning(
                        "%s failed (attempt %d/%d): %s - retrying in %.2fs",
                        fn.__name__, attempt, attempts, exc, delay,
                    )
                    time.sleep(delay)
            log.error("%s exhausted %d attempts: %s", fn.__name__, attempts, last)
            raise TransientError(f"{fn.__name__} failed after {attempts} attempts") from last

        return wrapper

    return decorator


# --------------------------------------------------------------------------
# Rate limiting
# --------------------------------------------------------------------------
class RateLimiter:
    """
    Sliding-window limiter. Thread-safe.

    Broker APIs publish hard per-second/per-minute caps and answer 429 when you
    exceed them; staying under the cap is cheaper than handling the rejection.
    """

    def __init__(self, max_calls: int, period_seconds: float) -> None:
        self.max_calls = max(1, int(max_calls))
        self.period = float(period_seconds)
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._calls and now - self._calls[0] > self.period:
                    self._calls.popleft()
                if len(self._calls) < self.max_calls:
                    self._calls.append(now)
                    return
                sleep_for = self.period - (now - self._calls[0]) + 0.01
            time.sleep(max(0.0, sleep_for))

    def __enter__(self) -> "RateLimiter":
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        return None


# --------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------
def chunked(seq: Sequence[T], size: int) -> Iterator[list[T]]:
    """Yield fixed-size chunks - broker quote endpoints are batch-limited."""
    size = max(1, int(size))
    for i in range(0, len(seq), size):
        yield list(seq[i: i + size])


class Timer:
    """Context manager that logs how long a block took."""

    def __init__(self, label: str, logger: logging.Logger | None = None, threshold_ms: float = 0.0):
        self.label = label
        self.log = logger or log
        self.threshold_ms = threshold_ms
        self.elapsed_ms = 0.0

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed_ms = (time.perf_counter() - self._t0) * 1000.0
        if self.elapsed_ms >= self.threshold_ms:
            self.log.debug("%s took %.1f ms", self.label, self.elapsed_ms)


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division that returns `default` instead of raising or producing inf/nan."""
    try:
        if denominator == 0 or denominator != denominator:
            return default
        result = numerator / denominator
        return result if result == result and abs(result) != float("inf") else default
    except (TypeError, ZeroDivisionError):
        return default


def pct_change(new: float, old: float) -> float:
    """Percentage change, guarded against a zero base."""
    return safe_div(new - old, abs(old), 0.0) * 100.0
