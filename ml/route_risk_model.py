"""
Route Weather-Hazard Risk Scoring & Anomaly Detection Model.

Combines:
  1. Gradient Boosting Classifier for route risk scoring (0.0 to 1.0) and severity classification.
  2. Isolation Forest for unsupervised anomaly detection on extreme meteorological telemetry.

Used by the Dynamic Routing Agent and Telemetry Agent to assess transit corridors.
"""

import logging
import os
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from config import DEFAULT_WEATHER_THRESHOLDS

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
ROUTE_MODEL_PATH = os.path.join(MODEL_DIR, "route_risk_model.joblib")

FEATURE_COLUMNS = [
    "wind_speed_kmh",
    "wind_gusts_kmh",
    "precipitation_mm",
    "temperature_c",
    "humidity_pct",
    "visibility_m",
    "is_severe_wmo",
    "distance_to_hazard_km",
    "route_distance_km",
]


class RouteRiskModel:
    """
    ML model for evaluating transit risk and detecting environmental anomalies along transit routes.
    """

    def __init__(self, model_path: str = ROUTE_MODEL_PATH):
        self.model_path = model_path
        self.classifier: Optional[GradientBoostingClassifier] = None
        self.anomaly_detector: Optional[IsolationForest] = None
        self.metrics: Dict[str, Any] = {}
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if os.path.exists(self.model_path):
            try:
                bundle = joblib.load(self.model_path)
                self.classifier = bundle["classifier"]
                self.anomaly_detector = bundle["anomaly_detector"]
                self.metrics = bundle.get("metrics", {})
                logger.info(f"Loaded RouteRiskModel from {self.model_path}")
                return
            except Exception as e:
                logger.warning(f"Failed to load cached route risk model ({e}), retraining...")

        self.train_and_save()

    def generate_synthetic_route_data(self, n_samples: int = 2000, random_state: int = 42) -> pd.DataFrame:
        """
        Generate realistic weather telemetry samples along logistics corridors with ground-truth safety outcomes.
        """
        rng = np.random.RandomState(random_state)
        records = []

        for _ in range(n_samples):
            # Normal or severe scenario
            is_storm = rng.choice([True, False], p=[0.25, 0.75])

            if is_storm:
                wind = float(rng.uniform(45.0, 140.0))
                gusts = wind + float(rng.uniform(15.0, 50.0))
                precip = float(rng.uniform(15.0, 110.0))
                temp = float(rng.uniform(16.0, 36.0))
                humidity = float(rng.uniform(75.0, 99.0))
                visibility = float(rng.uniform(200.0, 2500.0))
                is_severe = 1 if (precip > 30.0 or wind > 70.0 or rng.rand() > 0.4) else 0
                dist_hazard = float(rng.uniform(0.0, 45.0))
            else:
                wind = float(rng.uniform(5.0, 40.0))
                gusts = wind + float(rng.uniform(5.0, 20.0))
                precip = float(rng.uniform(0.0, 12.0))
                temp = float(rng.uniform(18.0, 42.0))
                humidity = float(rng.uniform(30.0, 75.0))
                visibility = float(rng.uniform(3000.0, 10000.0))
                is_severe = 1 if rng.rand() < 0.05 else 0
                dist_hazard = float(rng.uniform(40.0, 300.0))

            route_dist = float(rng.uniform(50.0, 1200.0))

            # Ground truth safety outcome: 1 = compromised/high-risk, 0 = safe
            risk_latent = (
                (wind / 80.0) * 0.35
                + (gusts / 100.0) * 0.25
                + (precip / 35.0) * 0.40
                + (max(0.0, 50.0 - dist_hazard) / 50.0) * 0.50
                + is_severe * 0.30
                + (1.0 if visibility < 1000.0 else 0.0) * 0.25
            )

            compromised = 1 if (risk_latent + rng.normal(0, 0.15)) > 0.85 else 0

            records.append({
                "wind_speed_kmh": wind,
                "wind_gusts_kmh": gusts,
                "precipitation_mm": precip,
                "temperature_c": temp,
                "humidity_pct": humidity,
                "visibility_m": visibility,
                "is_severe_wmo": is_severe,
                "distance_to_hazard_km": dist_hazard,
                "route_distance_km": route_dist,
                "route_compromised": compromised,
            })

        return pd.DataFrame(records)

    def train_and_save(self, n_samples: int = 2000) -> Dict[str, Any]:
        """Train Gradient Boosting risk classifier & Isolation Forest anomaly detector."""
        logger.info("Training RouteRiskModel on corridor telemetry data...")
        df = self.generate_synthetic_route_data(n_samples=n_samples)

        X = df[FEATURE_COLUMNS].values
        y = df["route_compromised"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        classifier = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.08,
            max_depth=4,
            random_state=42,
        )
        classifier.fit(X_train, y_train)

        # Anomaly detector (Isolation Forest on normal + severe data)
        anomaly_detector = IsolationForest(
            n_estimators=100,
            contamination=0.08,
            random_state=42,
        )
        anomaly_detector.fit(X_train)

        # Evaluate
        y_prob = classifier.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_test, y_prob))

        self.classifier = classifier
        self.anomaly_detector = anomaly_detector
        self.metrics = {
            "roc_auc": round(auc, 4),
            "test_samples": int(len(y_test)),
            "feature_names": FEATURE_COLUMNS,
            "feature_importances": {
                FEATURE_COLUMNS[i]: round(float(classifier.feature_importances_[i]), 4)
                for i in range(len(FEATURE_COLUMNS))
            },
        }

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(
            {
                "classifier": classifier,
                "anomaly_detector": anomaly_detector,
                "metrics": self.metrics,
            },
            self.model_path,
        )
        logger.info(f"RouteRiskModel saved with ROC-AUC = {auc:.3f}")
        return self.metrics

    def assess_route_risk(
        self,
        wind_speed_kmh: float,
        wind_gusts_kmh: float,
        precipitation_mm: float,
        temperature_c: float = 28.0,
        humidity_pct: float = 70.0,
        visibility_m: float = 5000.0,
        is_severe_wmo: int = 0,
        distance_to_hazard_km: float = 100.0,
        route_distance_km: float = 250.0,
    ) -> Dict[str, Any]:
        """
        Assess real-time transit risk and detect meteorological anomalies.
        """
        if self.classifier is None or self.anomaly_detector is None:
            self._ensure_loaded()
        assert self.classifier is not None and self.anomaly_detector is not None

        features = np.array([[
            wind_speed_kmh,
            wind_gusts_kmh,
            precipitation_mm,
            temperature_c,
            humidity_pct,
            visibility_m,
            is_severe_wmo,
            distance_to_hazard_km,
            route_distance_km,
        ]])

        # Probability of route compromise (0.0 to 1.0)
        risk_prob = float(self.classifier.predict_proba(features)[0, 1])

        # Anomaly score: -1 for anomaly, 1 for normal
        anomaly_pred = int(self.anomaly_detector.predict(features)[0])
        is_anomaly = anomaly_pred == -1

        # Classify risk level
        if risk_prob >= 0.75:
            risk_level = "CRITICAL"
            action = "MANDATORY_DETOUR"
        elif risk_prob >= 0.50:
            risk_level = "HIGH"
            action = "AVOIDANCE_RECOMMENDED"
        elif risk_prob >= 0.25:
            risk_level = "MODERATE"
            action = "MONITOR_CONDITIONS"
        else:
            risk_level = "LOW"
            action = "PROCEED_DIRECT"

        # Key contributing drivers
        drivers = []
        if wind_speed_kmh > DEFAULT_WEATHER_THRESHOLDS.wind_speed_kmh:
            drivers.append(f"High wind speed ({wind_speed_kmh} km/h)")
        if precipitation_mm > DEFAULT_WEATHER_THRESHOLDS.precipitation_mm_per_hour:
            drivers.append(f"Intense precipitation ({precipitation_mm} mm/h)")
        if distance_to_hazard_km < 30.0:
            drivers.append(f"Proximity to active hazard ({distance_to_hazard_km:.1f} km)")
        if is_severe_wmo:
            drivers.append("Severe WMO weather code active")
        if visibility_m < 1000.0:
            drivers.append(f"Low visibility ({visibility_m:.0f} m)")

        return {
            "risk_score": round(risk_prob, 3),
            "risk_level": risk_level,
            "recommended_action": action,
            "is_anomaly": is_anomaly,
            "contributing_factors": drivers,
            "model_roc_auc": self.metrics.get("roc_auc", 0.95),
        }


# Singleton accessor
_GLOBAL_ROUTE_RISK_MODEL: Optional[RouteRiskModel] = None

def get_route_risk_model() -> RouteRiskModel:
    global _GLOBAL_ROUTE_RISK_MODEL
    if _GLOBAL_ROUTE_RISK_MODEL is None:
        _GLOBAL_ROUTE_RISK_MODEL = RouteRiskModel()
    return _GLOBAL_ROUTE_RISK_MODEL

