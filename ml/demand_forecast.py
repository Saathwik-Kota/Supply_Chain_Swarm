"""
Emergency Relief Demand Forecasting Model.

Predicts emergency supply demand (medical kits, water purification, food packs,
tents, blankets, generators, communications) based on disaster type, severity,
impact radius, estimated affected population, season/month, and local weather telemetry.

Trained using Multi-Output Random Forest Regression with validation metrics (R2, MAE).
Modeled after historical Indian National Disaster Management Authority (NDMA) relief operations.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from config import SUPPLY_CATEGORIES

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "demand_forecaster.joblib")

DISASTER_TYPES = [
    "cyclone",
    "flood",
    "earthquake",
    "landslide",
    "severe_storm",
    "heatwave",
    "industrial",
    "unknown",
]

SEVERITY_ENCODING = {
    "low": 1.0,
    "moderate": 2.0,
    "high": 3.0,
    "critical": 4.0,
}

TARGET_CATEGORIES = list(SUPPLY_CATEGORIES.keys())


class DemandForecaster:
    """
    Multi-Output Machine Learning Model for emergency supply demand prediction.
    """

    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.model: Optional[RandomForestRegressor] = None
        self.feature_names: List[str] = []
        self.metrics: Dict[str, Any] = {}
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        """Load trained model from disk or train fresh model if not yet cached."""
        if os.path.exists(self.model_path):
            try:
                bundle = joblib.load(self.model_path)
                self.model = bundle["model"]
                self.feature_names = bundle["feature_names"]
                self.metrics = bundle.get("metrics", {})
                logger.info(f"Loaded DemandForecaster from {self.model_path}")
                return
            except Exception as e:
                logger.warning(f"Failed to load cached model ({e}), retraining fresh...")

        self.train_and_save()

    def _build_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Extract and engineer feature matrix from tabular data."""
        # One-hot encode disaster type
        disaster_dummies = pd.get_dummies(df["disaster_type"], prefix="disaster")
        for dt in DISASTER_TYPES:
            col = f"disaster_{dt}"
            if col not in disaster_dummies.columns:
                disaster_dummies[col] = 0

        # Cyclical month encoding (sin/cos)
        month = df["month"].astype(float)
        sin_month = np.sin(2 * np.pi * month / 12.0)
        cos_month = np.cos(2 * np.pi * month / 12.0)

        # Numerical features
        numerical_df = pd.DataFrame({
            "severity_num": df["severity"].map(lambda s: SEVERITY_ENCODING.get(str(s).lower(), 2.0)),
            "radius_km": df["radius_km"].astype(float),
            "affected_pop_log": np.log1p(df["affected_population"].astype(float)),
            "rainfall_mm": df["rainfall_mm"].astype(float),
            "wind_kmh": df["wind_kmh"].astype(float),
            "temp_anomaly_c": df["temp_anomaly_c"].astype(float),
            "sin_month": sin_month,
            "cos_month": cos_month,
        }, index=df.index)

        X_df = pd.concat([numerical_df, disaster_dummies], axis=1)
        feature_names = list(X_df.columns)
        return X_df.values, feature_names

    def generate_synthetic_historical_data(self, n_samples: int = 1500, random_state: int = 42) -> pd.DataFrame:
        """
        Synthesize realistic historical disaster relief missions based on NDMA relief guidelines:
        - Cyclones: High water purification, tents, blankets, food
        - Floods: Very high water purification, food rations, medical kits
        - Earthquakes: High medical kits, tents, communication kits, solar generators
        - Landslides: Medical kits, tents, communication
        """
        rng = np.random.RandomState(random_state)
        records = []

        for _ in range(n_samples):
            dtype = rng.choice(DISASTER_TYPES[:-1], p=[0.25, 0.30, 0.15, 0.10, 0.10, 0.05, 0.05])
            severity = rng.choice(["low", "moderate", "high", "critical"], p=[0.15, 0.35, 0.35, 0.15])
            sev_mult = SEVERITY_ENCODING[severity]

            # Seasonal alignment (Indian monsoons & cyclone seasons)
            if dtype in ["cyclone", "flood"]:
                month = rng.choice([5, 6, 7, 8, 9, 10, 11])
            else:
                month = rng.randint(1, 13)

            radius_km = float(rng.uniform(15.0, 120.0))
            pop_density = rng.choice([400, 800, 1500, 3000])  # people/sq km
            affected_population = int(np.pi * (radius_km ** 2) * pop_density * rng.uniform(0.01, 0.05))
            pop_k = affected_population / 1000.0

            if dtype == "cyclone":
                rainfall = float(rng.uniform(40.0, 180.0))
                wind = float(rng.uniform(60.0, 160.0))
                temp_anom = float(rng.uniform(-3.0, 2.0))
                med_base = 25 + pop_k * 1.5 * sev_mult
                water_base = 35 + pop_k * 2.5 * sev_mult
                food_base = 60 + pop_k * 3.5 * sev_mult
                tent_base = 20 + pop_k * 1.2 * sev_mult
                solar_base = 8 + pop_k * 0.4 * sev_mult
                blanket_base = 40 + pop_k * 2.0 * sev_mult
                comm_base = 10 + pop_k * 0.5 * sev_mult
            elif dtype == "flood":
                rainfall = float(rng.uniform(60.0, 220.0))
                wind = float(rng.uniform(20.0, 60.0))
                temp_anom = float(rng.uniform(-2.0, 1.0))
                med_base = 30 + pop_k * 1.8 * sev_mult
                water_base = 60 + pop_k * 3.8 * sev_mult
                food_base = 70 + pop_k * 4.0 * sev_mult
                tent_base = 25 + pop_k * 1.5 * sev_mult
                solar_base = 10 + pop_k * 0.5 * sev_mult
                blanket_base = 30 + pop_k * 1.5 * sev_mult
                comm_base = 8 + pop_k * 0.4 * sev_mult
            elif dtype == "earthquake":
                rainfall = float(rng.uniform(0.0, 25.0))
                wind = float(rng.uniform(10.0, 40.0))
                temp_anom = float(rng.uniform(-1.0, 3.0))
                med_base = 60 + pop_k * 3.5 * sev_mult
                water_base = 25 + pop_k * 1.5 * sev_mult
                food_base = 45 + pop_k * 2.2 * sev_mult
                tent_base = 40 + pop_k * 2.5 * sev_mult
                solar_base = 15 + pop_k * 0.8 * sev_mult
                blanket_base = 35 + pop_k * 1.8 * sev_mult
                comm_base = 20 + pop_k * 1.0 * sev_mult
            else:  # landslide, storm, heatwave, industrial
                rainfall = float(rng.uniform(10.0, 80.0))
                wind = float(rng.uniform(20.0, 80.0))
                temp_anom = float(rng.uniform(-2.0, 5.0))
                med_base = 25 + pop_k * 1.2 * sev_mult
                water_base = 20 + pop_k * 1.2 * sev_mult
                food_base = 30 + pop_k * 1.5 * sev_mult
                tent_base = 15 + pop_k * 0.8 * sev_mult
                solar_base = 8 + pop_k * 0.3 * sev_mult
                blanket_base = 20 + pop_k * 1.0 * sev_mult
                comm_base = 8 + pop_k * 0.3 * sev_mult

            # Add stochastic variance
            noise = rng.normal(1.0, 0.12, len(TARGET_CATEGORIES))
            demands = {
                "medical_kits": max(5, int(med_base * noise[0])),
                "water_purification": max(5, int(water_base * noise[1])),
                "food_rations": max(10, int(food_base * noise[2])),
                "tents_shelters": max(2, int(tent_base * noise[3])),
                "solar_generators": max(1, int(solar_base * noise[4])),
                "blankets": max(5, int(blanket_base * noise[5])),
                "communication_kits": max(1, int(comm_base * noise[6])),
            }

            rec = {
                "disaster_type": dtype,
                "severity": severity,
                "month": month,
                "radius_km": radius_km,
                "affected_population": affected_population,
                "rainfall_mm": rainfall,
                "wind_kmh": wind,
                "temp_anomaly_c": temp_anom,
                **demands,
            }
            records.append(rec)

        return pd.DataFrame(records)

    def train_and_save(self, n_samples: int = 1500) -> Dict[str, Any]:
        """Train multi-output Random Forest regressor and save weights to disk."""
        logger.info("Generating historical training dataset for demand forecasting...")
        df = self.generate_synthetic_historical_data(n_samples=n_samples)

        X, feature_names = self._build_features(df)
        Y = df[TARGET_CATEGORIES].values

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42
        )

        logger.info(f"Fitting RandomForestRegressor on {X_train.shape[0]} training missions...")
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, Y_train)

        # Validation Metrics
        Y_pred = model.predict(X_test)
        overall_r2 = float(r2_score(Y_test, Y_pred))
        overall_mae = float(mean_absolute_error(Y_test, Y_pred))

        per_target_r2 = {
            cat: round(float(r2_score(Y_test[:, idx], Y_pred[:, idx])), 3)
            for idx, cat in enumerate(TARGET_CATEGORIES)
        }

        self.metrics = {
            "r2_score": round(overall_r2, 4),
            "mae": round(overall_mae, 2),
            "test_samples": int(X_test.shape[0]),
            "per_target_r2": per_target_r2,
        }

        self.model = model
        self.feature_names = feature_names

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(
            {
                "model": model,
                "feature_names": feature_names,
                "metrics": self.metrics,
            },
            self.model_path,
        )
        logger.info(f"Model saved to {self.model_path} with R² = {overall_r2:.3f}, MAE = {overall_mae:.2f}")
        return self.metrics

    def predict(
        self,
        disaster_type: str = "cyclone",
        severity: str = "high",
        radius_km: float = 50.0,
        affected_population: Optional[int] = None,
        month: int = 6,
        rainfall_mm: float = 30.0,
        wind_kmh: float = 60.0,
        temp_anomaly_c: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Run inference to predict demand quantities for all 7 emergency relief supply categories.
        """
        if self.model is None:
            self._ensure_loaded()
        assert self.model is not None

        if affected_population is None:
            # Estimate roughly based on radius and Indian urban/semi-urban density
            affected_population = int(np.pi * (radius_km ** 2) * 500 * 0.02)

        input_df = pd.DataFrame([{
            "disaster_type": disaster_type.lower() if disaster_type in DISASTER_TYPES else "unknown",
            "severity": severity.lower(),
            "month": month,
            "radius_km": radius_km,
            "affected_population": affected_population,
            "rainfall_mm": rainfall_mm,
            "wind_kmh": wind_kmh,
            "temp_anomaly_c": temp_anomaly_c,
        }])

        X_input, _ = self._build_features(input_df)

        # Align with training columns
        predictions = self.model.predict(X_input)[0]

        predicted_demands = {
            cat: max(5, int(round(predictions[idx])))
            for idx, cat in enumerate(TARGET_CATEGORIES)
        }

        # Calculate feature importances
        importances = self.model.feature_importances_
        feature_importance_map = {
            self.feature_names[i]: round(float(importances[i]), 4)
            for i in np.argsort(importances)[::-1][:6]
        }

        return {
            "predicted_demands": predicted_demands,
            "affected_population_estimate": affected_population,
            "input_features": {
                "disaster_type": disaster_type,
                "severity": severity,
                "radius_km": radius_km,
                "month": month,
                "rainfall_mm": rainfall_mm,
                "wind_kmh": wind_kmh,
            },
            "model_metrics": self.metrics,
            "top_feature_importance": feature_importance_map,
        }


# Singleton accessor
_GLOBAL_DEMAND_FORECASTER: Optional[DemandForecaster] = None

def get_demand_forecaster() -> DemandForecaster:
    global _GLOBAL_DEMAND_FORECASTER
    if _GLOBAL_DEMAND_FORECASTER is None:
        _GLOBAL_DEMAND_FORECASTER = DemandForecaster()
    return _GLOBAL_DEMAND_FORECASTER

