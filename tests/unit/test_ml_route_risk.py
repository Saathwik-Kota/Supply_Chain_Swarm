"""
Unit tests for Machine Learning Route Risk & Anomaly Scoring Model.
"""

from ml.route_risk_model import get_route_risk_model


def test_route_risk_model_initialization():
    model = get_route_risk_model()
    assert model is not None
    assert model.classifier is not None
    assert model.anomaly_detector is not None


def test_mild_weather_evaluates_low_risk():
    model = get_route_risk_model()
    res = model.assess_route_risk(
        wind_speed_kmh=15.0,
        wind_gusts_kmh=22.0,
        precipitation_mm=0.0,
        temperature_c=25.0,
        visibility_m=8000.0,
        distance_to_hazard_km=150.0,
        route_distance_km=120.0,
    )
    assert res["risk_level"] == "LOW"
    assert res["risk_score"] < 0.35
    assert res["recommended_action"] == "PROCEED_DIRECT"
    assert isinstance(res["is_anomaly"], bool)


def test_severe_storm_evaluates_high_risk():
    model = get_route_risk_model()
    res = model.assess_route_risk(
        wind_speed_kmh=95.0,
        wind_gusts_kmh=130.0,
        precipitation_mm=55.0,
        temperature_c=24.0,
        visibility_m=600.0,
        is_severe_wmo=1,
        distance_to_hazard_km=8.0,
        route_distance_km=250.0,
    )
    assert res["risk_score"] > 0.40
    assert len(res["contributing_factors"]) > 0


def test_route_model_metrics():
    model = get_route_risk_model()
    assert "roc_auc" in model.metrics
    assert model.metrics["roc_auc"] >= 0.90

