"""
Unit tests for Change 14 (Async concurrent polling), Change 16 (Circuit breakers),
and Change 18 (TTL caches & Token Bucket rate limiters).
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.cache_and_ratelimit import (
    TokenBucketRateLimiter,
    TTLCache,
    clear_all_caches,
    get_all_cache_and_ratelimit_stats,
)
from services.circuit_breaker import (
    CircuitBreaker,
    get_all_breaker_statuses,
    register_circuit_breaker,
    reset_all_breakers,
)
from services.disaster_service import DisasterService
from services.routing_service import RoutingService, RoutingServiceError
from services.weather_service import WeatherService


@pytest.fixture(autouse=True)
def clean_resilience_state():
    reset_all_breakers()
    clear_all_caches()
    yield
    reset_all_breakers()
    clear_all_caches()


def test_circuit_breaker_open_half_open_closed_transitions():
    cb = register_circuit_breaker(
        "test_cb_unit",
        CircuitBreaker("test_cb_unit", failure_threshold=2, recovery_timeout=0.05),
    )
    assert cb.state == "CLOSED"
    assert cb.can_attempt() is True

    cb.record_failure()
    assert cb.state == "CLOSED"

    cb.record_failure()
    assert cb.state == "OPEN"
    assert cb.can_attempt() is False

    time.sleep(0.08)
    assert cb.can_attempt() is True
    assert cb.state == "HALF-OPEN"

    cb.record_success()
    assert cb.state == "CLOSED"

    statuses = get_all_breaker_statuses()
    assert "test_cb_unit" in statuses
    assert statuses["test_cb_unit"]["total_short_circuits"] >= 1


def test_ttl_cache_hit_miss_lru_and_expiry():
    cache = TTLCache("unit_cache", ttl_seconds=0.1, max_size=2)
    assert cache.get("missing") is None

    cache.set("k1", {"v": 1})
    cache.set("k2", {"v": 2})
    assert cache.get("k1") == {"v": 1}  # bumps k1 to most-recently-used

    cache.set("k3", {"v": 3})  # evicts k2
    assert cache.get("k2") is None
    assert cache.get("k1") == {"v": 1}
    assert cache.get("k3") == {"v": 3}

    time.sleep(0.15)
    assert cache.get("k1") is None
    stats = cache.stats()
    assert stats["hits"] >= 3
    assert stats["misses"] >= 3


def test_token_bucket_rate_limiter_consume_and_refill():
    limiter = TokenBucketRateLimiter("unit_limiter", max_calls=2, period_seconds=0.1)
    assert limiter.acquire(wait=False) is True
    assert limiter.acquire(wait=False) is True
    assert limiter.acquire(wait=False) is False

    # Blocking wait refills tokens quickly at 20 req/sec (~50ms per token)
    assert limiter.acquire(wait=True, timeout=1.0) is True


def test_weather_async_corridor_scan_and_caching():
    svc = WeatherService()
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.status_code = 200
    mock_resp.url = "https://api.open-meteo.com/v1/forecast"
    mock_resp.json.return_value = {
        "current": {
            "temperature_2m": 31.0,
            "wind_speed_10m": 18.0,
            "wind_gusts_10m": 28.0,
            "precipitation": 0.5,
            "rain": 0.5,
            "weather_code": 1,
            "relative_humidity_2m": 65,
            "time": "2026-11-11T00:00",
        }
    }

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("services.weather_service.httpx.AsyncClient", return_value=mock_client):
        scan1 = asyncio.run(svc.scan_all_corridors_async())
        assert scan1["total_hubs_scanned"] == 20
        assert len(scan1["hubs"]) == 20
        assert scan1["hubs"][0]["weather"]["temperature_c"] == 31.0

        # Second scan hits TTL cache for all 20 hubs
        scan2 = asyncio.run(svc.scan_all_corridors_async())
        assert scan2["total_hubs_scanned"] == 20
        assert scan2["cache_hits"] == 20


def test_disaster_service_async_concurrent_feeds():
    svc = DisasterService()
    usgs_resp = MagicMock()
    usgs_resp.raise_for_status.return_value = None
    usgs_resp.json.return_value = {
        "features": [
            {
                "properties": {"mag": 5.4, "place": "Andaman Islands, India", "time": 1700000000},
                "geometry": {"coordinates": [92.7, 11.7, 10.0]},
            }
        ]
    }
    eonet_resp = MagicMock()
    eonet_resp.raise_for_status.return_value = None
    eonet_resp.json.return_value = {
        "events": [
            {
                "id": "EONET_TEST_1",
                "title": "Cyclone Test Bay of Bengal",
                "categories": [{"title": "Severe Storms"}],
                "geometry": [{"coordinates": [81.5, 13.5], "date": "2026-11-11T00:00:00Z"}],
            }
        ]
    }

    async def fake_get(url, *args, **kwargs):
        return usgs_resp if "usgs.gov" in url else eonet_resp

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=fake_get)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("services.disaster_service.httpx.AsyncClient", return_value=mock_client):
        hazards = asyncio.run(svc.get_regional_hazards_async())
        assert len(hazards) == 2
        sources = {h["source"] for h in hazards}
        assert "USGS Live Seismic Feed" in sources
        assert "NASA EONET v3 Live Feed" in sources


def test_routing_service_cache_and_circuit_breaker():
    rsvc = RoutingService()
    rsvc._breaker.reset()
    rsvc._cache.clear()
    mock_http_resp = MagicMock()
    mock_http_resp.raise_for_status.return_value = None
    mock_http_resp.url = "https://router.project-osrm.org/route/v1/driving/test"
    mock_http_resp.json.return_value = {
        "code": "Ok",
        "routes": [{"distance": 345000.0, "duration": 21600.0, "geometry": "oednAohqxMonT_mjO"}],
    }

    try:
        with patch.object(rsvc._client, "get", return_value=mock_http_resp) as mock_get:
            r1 = rsvc.get_route(12.97, 77.59, 13.08, 80.27)
            r2 = rsvc.get_route(12.97, 77.59, 13.08, 80.27)
            assert r1["distance_km"] == r2["distance_km"]
            assert mock_get.call_count == 1
            assert rsvc._cache.stats()["hits"] >= 1

        # Trip OSRM circuit breaker via failure threshold and verify fast-fail error + fallback
        for _ in range(rsvc._breaker.failure_threshold):
            rsvc._breaker.record_failure()
        assert rsvc._breaker.state == "OPEN"

        with pytest.raises(RoutingServiceError):
            rsvc._query_osrm([(72.87, 19.07), (73.85, 18.52)])
        fallback = rsvc.get_route(19.07, 72.87, 18.52, 73.85)
        assert fallback["_meta"]["source"] == "geodesic_offline_fallback"
        stats = get_all_cache_and_ratelimit_stats()
        assert "routing" in stats["caches"]
    finally:
        rsvc._breaker.reset()
