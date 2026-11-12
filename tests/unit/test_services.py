"""
Unit tests for external service wrappers and geometric routing math.
"""

from config import WeatherThresholds
from services.routing_service import (
    create_hazard_circle,
    haversine_distance,
    offset_point,
)
from services.weather_service import WeatherCircuitBreaker, WeatherService


def test_haversine_distance_known_corridor():
    # Bangalore (12.98, 77.75) to Chennai (13.12, 80.29)
    dist = haversine_distance(12.98, 77.75, 13.12, 80.29)
    # Great circle distance is ~275 - 290 km
    assert 260.0 < dist < 305.0


def test_offset_point():
    origin_lat, origin_lon = 13.0, 80.0
    # Move 50 km North (bearing 0)
    new_lat, new_lon = offset_point(origin_lat, origin_lon, bearing_deg=0.0, distance_km=50.0)
    assert new_lat > origin_lat
    assert abs(new_lon - origin_lon) < 0.05


def test_hazard_assessment_logic():
    ws = WeatherService()
    thresholds = WeatherThresholds(
        wind_speed_kmh=80.0,
        precipitation_mm_per_hour=30.0,
    )

    safe_weather = {"wind_speed_kmh": 20.0, "precipitation_mm": 5.0, "temperature_c": 28.0}
    safe_eval = ws.assess_hazard(safe_weather, thresholds)
    assert safe_eval["is_hazardous"] is False

    cyclone_weather = {"wind_speed_kmh": 110.0, "precipitation_mm": 50.0, "temperature_c": 26.0}
    storm_eval = ws.assess_hazard(cyclone_weather, thresholds)
    assert storm_eval["is_hazardous"] is True
    assert storm_eval["severity"] in ["high", "critical"]


def test_weather_circuit_breaker():
    cb = WeatherCircuitBreaker(failure_threshold=2, recovery_timeout=0.5)
    assert cb.state == "CLOSED"
    assert cb.can_attempt() is True

    # 1st failure
    cb.record_failure()
    assert cb.state == "CLOSED"

    # 2nd failure -> OPEN
    cb.record_failure()
    assert cb.state == "OPEN"
    assert cb.can_attempt() is False

    # Successful recovery
    cb.record_success()
    assert cb.state == "CLOSED"
    assert cb.can_attempt() is True


def test_hazard_polygon_creation():
    poly = create_hazard_circle(13.0, 80.0, radius_km=50.0)
    assert poly.is_valid
    assert not poly.is_empty
    # Center point should be within polygon
    from shapely.geometry import Point
    assert poly.contains(Point(80.0, 13.0))

