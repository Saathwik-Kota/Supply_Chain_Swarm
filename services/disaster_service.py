"""
Live natural hazard and disaster telemetry service.
Integrates live USGS Earthquake API and NASA EONET (Earth Observatory Natural Event Tracker).

Features:
  - Async concurrent feed polling (USGS + NASA EONET in parallel)
  - Circuit breaker protection for external resilience
  - TTL caching and rate limiting
Zero API keys required — immediate real-time execution.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import httpx

from config import API_TIMEOUT_SECONDS, DISASTER_CACHE_TTL_SECONDS
from services.cache_and_ratelimit import get_rate_limiter, get_ttl_cache
from services.circuit_breaker import get_circuit_breaker

logger = logging.getLogger(__name__)

USGS_ALL_DAY_URL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson"
USGS_SIGNIFICANT_MONTH_URL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/significant_month.geojson"
NASA_EONET_EVENTS_URL = "https://eonet.gsfc.nasa.gov/api/v3/events"


def _parse_usgs_features(data: Dict[str, Any], min_magnitude: float) -> List[Dict[str, Any]]:
    """Transform USGS GeoJSON features into normalized disaster event dicts."""
    events = []
    for feat in data.get("features", []):
        props = feat.get("properties", {})
        coords = feat.get("geometry", {}).get("coordinates", [])
        mag = props.get("mag") or 0.0
        if mag >= min_magnitude and len(coords) >= 2:
            lon, lat = coords[0], coords[1]
            events.append({
                "id": feat.get("id"),
                "title": props.get("title", "Earthquake"),
                "hazard_type": "earthquake",
                "severity": "critical" if mag >= 6.0 else "high",
                "magnitude": mag,
                "lat": lat,
                "lon": lon,
                "depth_km": coords[2] if len(coords) > 2 else 10.0,
                "place": props.get("place", "Unknown"),
                "time": props.get("time"),
                "source": "USGS Live Seismic Feed",
                "radius_km": max(30.0, mag * 12.0),
            })
    return events


def _parse_eonet_events(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Transform NASA EONET v3 payload into normalized disaster event dicts."""
    hazards = []
    for ev in data.get("events", []):
        categories = [c.get("title", "") for c in ev.get("categories", [])]
        cat_title = categories[0] if categories else "Severe Weather"
        geoms = ev.get("geometry", [])
        if not geoms:
            continue
        latest_geom = geoms[-1]
        coords = latest_geom.get("coordinates", [])
        if isinstance(coords, list) and len(coords) >= 2 and isinstance(coords[0], (int, float)):
            lon, lat = coords[0], coords[1]
            hazards.append({
                "id": ev.get("id"),
                "title": ev.get("title"),
                "hazard_type": cat_title.lower().replace(" ", "_"),
                "category": cat_title,
                "severity": "high",
                "lat": lat,
                "lon": lon,
                "date": latest_geom.get("date"),
                "source": "NASA EONET v3 Live Feed",
                "radius_km": 60.0,
            })
    return hazards


class DisasterService:
    """Live disaster feed consumer for USGS and NASA EONET with circuit breakers & caching."""

    def __init__(self, timeout: float = API_TIMEOUT_SECONDS):
        self.timeout = timeout
        self._client = httpx.Client(timeout=self.timeout)
        self._usgs_breaker = get_circuit_breaker("usgs")
        self._eonet_breaker = get_circuit_breaker("nasa_eonet")
        self._cache = get_ttl_cache("disaster", ttl_seconds=DISASTER_CACHE_TTL_SECONDS)
        self._rate_limiter = get_rate_limiter("disaster_feeds")

    def close(self):
        self._client.close()

    def fetch_live_usgs_earthquakes(
        self, min_magnitude: float = 4.0, use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """Fetch LIVE earthquakes from USGS GeoJSON feed."""
        cache_key = f"usgs:{round(min_magnitude, 1)}"
        if use_cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return list(cached)

        if not self._usgs_breaker.can_attempt():
            logger.info("[CircuitBreaker:usgs] OPEN; skipping USGS feed call.")
            return []
        if not self._rate_limiter.acquire(wait=True, timeout=1.0):
            return []

        try:
            resp = self._client.get(USGS_ALL_DAY_URL)
            resp.raise_for_status()
            self._usgs_breaker.record_success()
            events = _parse_usgs_features(resp.json(), min_magnitude)
            if use_cache:
                self._cache.set(cache_key, events)
            return events
        except Exception as e:
            self._usgs_breaker.record_failure()
            logger.warning(f"USGS live earthquake fetch error: {e}")
            return []

    def fetch_live_nasa_eonet_hazards(self, use_cache: bool = True) -> List[Dict[str, Any]]:
        """Fetch LIVE active natural hazards from NASA EONET v3."""
        cache_key = "eonet:open:25"
        if use_cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return list(cached)

        if not self._eonet_breaker.can_attempt():
            logger.info("[CircuitBreaker:nasa_eonet] OPEN; skipping NASA EONET call.")
            return []
        if not self._rate_limiter.acquire(wait=True, timeout=1.0):
            return []

        try:
            resp = self._client.get(NASA_EONET_EVENTS_URL, params={"status": "open", "limit": 25})
            resp.raise_for_status()
            self._eonet_breaker.record_success()
            hazards = _parse_eonet_events(resp.json())
            if use_cache:
                self._cache.set(cache_key, hazards)
            return hazards
        except Exception as e:
            self._eonet_breaker.record_failure()
            logger.warning(f"NASA EONET live fetch error: {e}")
            return []

    async def fetch_live_usgs_earthquakes_async(
        self,
        min_magnitude: float = 4.0,
        async_client: Optional[httpx.AsyncClient] = None,
        use_cache: bool = True,
    ) -> List[Dict[str, Any]]:
        cache_key = f"usgs:{round(min_magnitude, 1)}"
        if use_cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return list(cached)

        if not self._usgs_breaker.can_attempt() or not self._rate_limiter.acquire(wait=True, timeout=1.0):
            return []

        owns = async_client is None
        client = async_client or httpx.AsyncClient(timeout=self.timeout)
        try:
            resp = await client.get(USGS_ALL_DAY_URL)
            resp.raise_for_status()
            self._usgs_breaker.record_success()
            events = _parse_usgs_features(resp.json(), min_magnitude)
            if use_cache:
                self._cache.set(cache_key, events)
            return events
        except Exception as e:
            self._usgs_breaker.record_failure()
            logger.warning(f"USGS async earthquake fetch error: {e}")
            return []
        finally:
            if owns:
                await client.aclose()

    async def fetch_live_nasa_eonet_hazards_async(
        self,
        async_client: Optional[httpx.AsyncClient] = None,
        use_cache: bool = True,
    ) -> List[Dict[str, Any]]:
        cache_key = "eonet:open:25"
        if use_cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return list(cached)

        if not self._eonet_breaker.can_attempt() or not self._rate_limiter.acquire(wait=True, timeout=1.0):
            return []

        owns = async_client is None
        client = async_client or httpx.AsyncClient(timeout=self.timeout)
        try:
            resp = await client.get(NASA_EONET_EVENTS_URL, params={"status": "open", "limit": 25})
            resp.raise_for_status()
            self._eonet_breaker.record_success()
            hazards = _parse_eonet_events(resp.json())
            if use_cache:
                self._cache.set(cache_key, hazards)
            return hazards
        except Exception as e:
            self._eonet_breaker.record_failure()
            logger.warning(f"NASA EONET async fetch error: {e}")
            return []
        finally:
            if owns:
                await client.aclose()

    async def get_regional_hazards_async(
        self,
        min_lat: float = 6.0,
        max_lat: float = 38.0,
        min_lon: float = 68.0,
        max_lon: float = 98.0,
    ) -> List[Dict[str, Any]]:
        """Concurrently fetch USGS and NASA EONET feeds filtered to India."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            eqs, nasa_events = await asyncio.gather(
                self.fetch_live_usgs_earthquakes_async(min_magnitude=3.5, async_client=client),
                self.fetch_live_nasa_eonet_hazards_async(async_client=client),
            )

        return [
            ev for ev in (eqs + nasa_events)
            if ev.get("lat") and ev.get("lon")
            and (min_lat <= ev["lat"] <= max_lat)
            and (min_lon <= ev["lon"] <= max_lon)
        ]

    def get_regional_hazards(
        self,
        min_lat: float = 6.0,
        max_lat: float = 38.0,
        min_lon: float = 68.0,
        max_lon: float = 98.0,
    ) -> List[Dict[str, Any]]:
        """Fetch live disasters concurrently via asyncio."""
        try:
            asyncio.get_running_loop()
            with ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(
                    lambda: asyncio.run(
                        self.get_regional_hazards_async(min_lat, max_lat, min_lon, max_lon)
                    )
                ).result()
        except RuntimeError:
            return asyncio.run(
                self.get_regional_hazards_async(min_lat, max_lat, min_lon, max_lon)
            )


_disaster_service: Optional[DisasterService] = None


def get_disaster_service() -> DisasterService:
    global _disaster_service
    if _disaster_service is None:
        _disaster_service = DisasterService()
    return _disaster_service
