"""
Machine Learning module for Autonomous Emergency Response & Supply Chain Logistics Swarm.

Provides:
  - Emergency relief demand forecasting (Random Forest / Gradient Boosting Multi-Output Regression)
  - Route weather-hazard risk scoring & anomaly detection (Isolation Forest + Classifier)
"""

from ml.demand_forecast import DemandForecaster, get_demand_forecaster
from ml.route_risk_model import RouteRiskModel, get_route_risk_model

__all__ = [
    "DemandForecaster",
    "get_demand_forecaster",
    "RouteRiskModel",
    "get_route_risk_model",
]

