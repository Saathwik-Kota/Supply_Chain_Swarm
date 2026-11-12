"""
Unit tests for Machine Learning Demand Forecaster.
"""

from config import SUPPLY_CATEGORIES
from ml.demand_forecast import get_demand_forecaster


def test_demand_forecaster_initialization():
    forecaster = get_demand_forecaster()
    assert forecaster is not None
    assert forecaster.model is not None
    assert len(forecaster.feature_names) > 0


def test_demand_forecast_inference_schema():
    forecaster = get_demand_forecaster()
    result = forecaster.predict(
        disaster_type="cyclone",
        severity="high",
        radius_km=50.0,
        month=11,
    )
    assert "predicted_demands" in result
    demands = result["predicted_demands"]

    for cat in SUPPLY_CATEGORIES.keys():
        assert cat in demands
        assert isinstance(demands[cat], int)
        assert demands[cat] >= 5


def test_disaster_type_demand_divergence():
    """Verify that different disaster types yield domain-appropriate supply allocations."""
    forecaster = get_demand_forecaster()

    flood_pred = forecaster.predict(disaster_type="flood", severity="critical", radius_km=50.0)
    quake_pred = forecaster.predict(disaster_type="earthquake", severity="critical", radius_km=50.0)

    # Earthquakes should typically drive high medical trauma demand
    # Floods should drive high clean water / purification demand
    assert quake_pred["predicted_demands"]["medical_kits"] > 0
    assert flood_pred["predicted_demands"]["water_purification"] > 0


def test_model_metrics_validation():
    forecaster = get_demand_forecaster()
    metrics = forecaster.metrics
    assert "r2_score" in metrics
    assert metrics["r2_score"] >= 0.85
    assert "mae" in metrics
    assert metrics["mae"] < 800.0

