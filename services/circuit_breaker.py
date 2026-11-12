"""
Thread-safe Circuit Breaker state machine for external service resilience.

Prevents cascading failures and retry storms when external APIs (Open-Meteo,
OSRM, USGS Seismic Feed, NASA EONET) become unreachable or degraded.
States:
  - CLOSED: Normal operation; failures increment counter.
  - OPEN: Fast-fail without external network calls until recovery_timeout elapses.
  - HALF-OPEN: Allows a single probe attempt; success closes, failure re-opens.
"""

import logging
import threading
import time
from typing import Any, Dict

from config import CIRCUIT_BREAKER_RECOVERY_SECONDS, CIRCUIT_BREAKER_THRESHOLD

logger = logging.getLogger(__name__)


class CircuitBreakerOpenError(Exception):
    """Raised when an operation is rejected because its circuit breaker is OPEN."""
    pass


class CircuitBreaker:
    """Thread-safe circuit breaker with metrics tracking."""

    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = CIRCUIT_BREAKER_THRESHOLD,
        recovery_timeout: float = CIRCUIT_BREAKER_RECOVERY_SECONDS,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.success_count = 0
        self.total_failures = 0
        self.total_short_circuits = 0
        self.last_failure_time = 0.0
        self.state = "CLOSED"
        self._lock = threading.Lock()

    def can_attempt(self) -> bool:
        with self._lock:
            if self.state == "CLOSED":
                return True
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = "HALF-OPEN"
                logger.info(f"[CircuitBreaker:{self.name}] Recovery window elapsed; entering HALF-OPEN.")
                return True
            self.total_short_circuits += 1
            return False

    def record_success(self) -> None:
        with self._lock:
            self.failure_count = 0
            self.success_count += 1
            if self.state != "CLOSED":
                logger.info(f"[CircuitBreaker:{self.name}] Probe succeeded; state restored to CLOSED.")
            self.state = "CLOSED"

    def record_failure(self) -> None:
        with self._lock:
            self.failure_count += 1
            self.total_failures += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                if self.state != "OPEN":
                    logger.warning(
                        f"[CircuitBreaker:{self.name}] Failure threshold ({self.failure_threshold}) "
                        f"reached. State set to OPEN for {self.recovery_timeout}s."
                    )
                self.state = "OPEN"

    def reset(self) -> None:
        with self._lock:
            self.failure_count = 0
            self.last_failure_time = 0.0
            self.state = "CLOSED"

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "name": self.name,
                "state": self.state,
                "failure_count": self.failure_count,
                "failure_threshold": self.failure_threshold,
                "recovery_timeout_s": self.recovery_timeout,
                "success_count": self.success_count,
                "total_failures": self.total_failures,
                "total_short_circuits": self.total_short_circuits,
            }


_BREAKER_REGISTRY: Dict[str, CircuitBreaker] = {}
_REGISTRY_LOCK = threading.Lock()


def get_circuit_breaker(
    name: str,
    failure_threshold: int = CIRCUIT_BREAKER_THRESHOLD,
    recovery_timeout: float = CIRCUIT_BREAKER_RECOVERY_SECONDS,
) -> CircuitBreaker:
    """Get or create a named singleton circuit breaker."""
    with _REGISTRY_LOCK:
        if name not in _BREAKER_REGISTRY:
            _BREAKER_REGISTRY[name] = CircuitBreaker(
                name=name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
            )
        return _BREAKER_REGISTRY[name]


def register_circuit_breaker(name: str, breaker: CircuitBreaker) -> CircuitBreaker:
    """Register an existing circuit breaker instance under a name."""
    with _REGISTRY_LOCK:
        _BREAKER_REGISTRY[name] = breaker
        return breaker


def get_all_breaker_statuses() -> Dict[str, Dict[str, Any]]:
    """Return status telemetry for all registered circuit breakers."""
    with _REGISTRY_LOCK:
        return {name: cb.get_status() for name, cb in _BREAKER_REGISTRY.items()}


def reset_all_breakers() -> None:
    """Reset all circuit breakers back to CLOSED."""
    with _REGISTRY_LOCK:
        for cb in _BREAKER_REGISTRY.values():
            cb.reset()

