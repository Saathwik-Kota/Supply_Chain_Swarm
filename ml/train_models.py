"""
ML Training Pipeline Script.

Trains:
  1. DemandForecaster (Multi-Output RandomForestRegressor on emergency relief datasets)
  2. RouteRiskModel (GradientBoostingClassifier + IsolationForest anomaly detector)

Run directly with:
    python ml/train_models.py
"""

import logging

from ml.demand_forecast import DemandForecaster
from ml.route_risk_model import RouteRiskModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("STARTING ML MODEL TRAINING PIPELINE")
    logger.info("=" * 60)

    # 1. Train Demand Forecaster
    logger.info("\n>>> [1/2] Training Emergency Relief Demand Forecaster...")
    forecaster = DemandForecaster()
    demand_metrics = forecaster.train_and_save(n_samples=2500)
    print("\n--- Demand Forecaster Evaluation ---")
    print(f"Overall R² Score : {demand_metrics.get('r2_score')}")
    print(f"Overall MAE      : {demand_metrics.get('mae')} units")
    print(f"Test Set Size    : {demand_metrics.get('test_samples')} samples")
    print("Per-Category R² Scores:")
    for cat, r2 in demand_metrics.get("per_target_r2", {}).items():
        print(f"  • {cat:<22}: {r2:.3f}")

    # Sample inference
    sample_pred = forecaster.predict(
        disaster_type="cyclone",
        severity="critical",
        radius_km=75.0,
        month=11,
        rainfall_mm=85.0,
        wind_kmh=120.0,
    )
    print("\nSample Cyclone Demand Inference:")
    for cat, qty in sample_pred["predicted_demands"].items():
        print(f"  • {cat:<22}: {qty} units")

    # 2. Train Route Risk Model
    logger.info("\n>>> [2/2] Training Route Risk & Hazard Anomaly Model...")
    risk_model = RouteRiskModel()
    risk_metrics = risk_model.train_and_save(n_samples=3000)
    print("\n--- Route Risk Model Evaluation ---")
    print(f"ROC-AUC Score    : {risk_metrics.get('roc_auc')}")
    print(f"Test Set Size    : {risk_metrics.get('test_samples')} samples")
    print("Top Feature Importances:")
    for feat, imp in risk_metrics.get("feature_importances", {}).items():
        print(f"  • {feat:<22}: {imp:.4f}")

    sample_risk = risk_model.assess_route_risk(
        wind_speed_kmh=85.0,
        wind_gusts_kmh=110.0,
        precipitation_mm=45.0,
        distance_to_hazard_km=12.0,
        route_distance_km=320.0,
        is_severe_wmo=1,
    )
    print("\nSample Storm Corridor Assessment:")
    print(f"  Risk Probability     : {sample_risk['risk_score'] * 100:.1f}%")
    print(f"  Risk Level           : {sample_risk['risk_level']}")
    print(f"  Recommended Action   : {sample_risk['recommended_action']}")
    print(f"  Meteorological Anomaly: {sample_risk['is_anomaly']}")
    print(f"  Contributing Factors : {', '.join(sample_risk['contributing_factors'])}")

    logger.info("\n" + "=" * 60)
    logger.info("ALL ML MODELS TRAINED & SERIALIZED SUCCESSFULLY")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

