"""
Thread-safe In-Memory TTL Cache & Token-Bucket Rate Limiter for External APIs.

Eliminates redundant Open-Meteo, USGS/NASA EONET, and OSRM HTTP calls across
rapid repeated swarm launches while preventing external API quota throttling.
"""

import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

from config import RATE_LIMIT_MAX_CALLS, RATE_LIMIT_PERIOD_SECONDS

logger = logging.getLogger(__name__)


class TTLCache:
    """Thread-safe LRU + TTL in-memory cache with hit/miss telemetry."""

    def __init__(self, name: str = "default", ttl_seconds: float = 300.0, max_size: int = 512):
        self.name = name
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._store: "OrderedDict[str, Tuple[float, Any]]" = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        now = time.time()
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self.misses += 1
                return None
            expires_at, value = entry
            if now >= expires_at:
                del self._store[key]
                self.misses += 1
                return None
            self._store.move_to_end(key)
            self.hits += 1
            return value

    def set(self, key: str, value: Any, ttl_seconds: Optional[float] = None) -> None:
        ttl = self.ttl_seconds if ttl_seconds is None else ttl_seconds
        expires_at = time.time() + ttl
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (expires_at, value)
            while len(self._store) > self.max_size:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self.hits = 0
            self.misses = 0

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self.hits + self.misses
            now = time.time()
            active_entries = sum(1 for exp, _ in self._store.values() if exp > now)
            return {
                "name": self.name,
                "ttl_seconds": self.ttl_seconds,
                "active_entries": active_entries,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate_pct": round((self.hits / total) * 100.0, 2) if total > 0 else 0.0,
            }


class TokenBucketRateLimiter:
    """Thread-safe Token Bucket rate limiter for outbound API protection."""

    def __init__(
        self,
        name: str = "default",
        max_calls: int = RATE_LIMIT_MAX_CALLS,
        period_seconds: float = RATE_LIMIT_PERIOD_SECONDS,
    ):
        self.name = name
        self.capacity = float(max(1, max_calls))
        self.refill_rate = self.capacity / max(0.001, float(period_seconds))
        self.tokens = self.capacity
        self.last_refill = time.time()
        self.total_allowed = 0
        self.total_throttled = 0
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = time.time()
        elapsed = max(0.0, now - self.last_refill)
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    def acquire(self, wait: bool = False, timeout: float = 5.0) -> bool:
        """Consume one request token; optionally block up to `timeout` seconds."""
        deadline = time.time() + timeout
        while True:
            with self._lock:
                self._refill()
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    self.total_allowed += 1
                    return True
                if not wait or time.time() >= deadline:
                    self.total_throttled += 1
                    return False
                wait_s = min(0.1, (1.0 - self.tokens) / self.refill_rate)
            time.sleep(max(0.01, wait_s))

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            self._refill()
            return {
                "name": self.name,
                "available_tokens": round(self.tokens, 2),
                "capacity": int(self.capacity),
                "total_allowed": self.total_allowed,
                "total_throttled": self.total_throttled,
            }


_CACHES: Dict[str, TTLCache] = {}
_RATE_LIMITERS: Dict[str, TokenBucketRateLimiter] = {}
_LOCK = threading.Lock()


def get_ttl_cache(name: str, ttl_seconds: int = 300, max_size: int = 512) -> TTLCache:
    with _LOCK:
        if name not in _CACHES:
            _CACHES[name] = TTLCache(name=name, ttl_seconds=ttl_seconds, max_size=max_size)
        return _CACHES[name]


def get_rate_limiter(
    name: str,
    max_calls: int = RATE_LIMIT_MAX_CALLS,
    period_seconds: float = RATE_LIMIT_PERIOD_SECONDS,
) -> TokenBucketRateLimiter:
    with _LOCK:
        if name not in _RATE_LIMITERS:
            _RATE_LIMITERS[name] = TokenBucketRateLimiter(
                name=name, max_calls=max_calls, period_seconds=period_seconds
            )
        return _RATE_LIMITERS[name]


def get_all_cache_and_ratelimit_stats() -> Dict[str, Any]:
    with _LOCK:
        return {
            "caches": {k: c.stats() for k, c in _CACHES.items()},
            "rate_limiters": {k: rl.stats() for k, rl in _RATE_LIMITERS.items()},
        }


def clear_all_caches() -> None:
    with _LOCK:
        for c in _CACHES.values():
            c.clear()

