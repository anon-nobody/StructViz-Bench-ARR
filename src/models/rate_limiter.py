from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field


@dataclass(slots=True)
class RateLimiter:
    """Thread-safe token bucket limiter with sync and async waiting."""

    requests_per_minute: int = 60
    _tokens: float = field(init=False)
    _last_refill: float = field(init=False)
    _lock: threading.Lock = field(init=False)

    def __post_init__(self) -> None:
        if self.requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be greater than zero")
        self._tokens = float(self.requests_per_minute)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    @property
    def _tokens_per_second(self) -> float:
        return self.requests_per_minute / 60.0

    def _refill(self, now: float) -> None:
        elapsed = now - self._last_refill
        self._tokens = min(
            float(self.requests_per_minute),
            self._tokens + (elapsed * self._tokens_per_second),
        )
        self._last_refill = now

    def _acquire_token(self) -> bool:
        with self._lock:
            now = time.monotonic()
            self._refill(now)
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            return False

    def wait(self) -> None:
        """Block until one token is available."""
        while not self._acquire_token():
            time.sleep(0.01)

    async def acquire(self) -> None:
        """Await until one token is available."""
        while not self._acquire_token():
            await asyncio.sleep(0.01)
