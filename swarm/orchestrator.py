"""
Multi-Agent Swarm Orchestrator Engine.

Coordinates the state graph execution of 4 autonomous agents:
  - Telemetry & Threat Monitor Agent (Agent 1)
  - Regional Inventory & Supplier Agent (Agent 2)
  - Dynamic Routing & Avoidance Agent (Agent 3)
  - Autonomous Dispatch & Fleet Coordinator (Agent 4)

Driven by LLM reasoning, ReAct decision loops, and Machine Learning intelligence
(Random Forest Demand Forecaster + Gradient Boosting Route Risk Anomaly detector).
"""

import logging
from typing import Any, Callable, Dict, Optional

from agents.dispatch_agent import run_dispatch_agent
from agents.inventory_agent import run_inventory_agent
from agents.routing_agent import run_routing_agent
from agents.telemetry_agent import run_telemetry_agent
from config import (
    CO2_KG_PER_LITER,
    FUEL_CONSUMPTION_L_PER_100KM,
    FUEL_COST_INR_PER_LITER,
    VEHICLE_TYPES,
)
from swarm.agent_framework import MultiAgentGraph
from swarm.state import SwarmState, create_initial_state

logger = logging.getLogger(__name__)


def compute_carbon_metrics(
    distance_km: float,
    vehicle_type: str,
    primary_distance_km: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute fuel consumption, CO₂ emissions, and cost for a given route.
    If primary_distance_km is given (detour case), also computes the detour overhead.
    """
    fuel_l_per_100 = FUEL_CONSUMPTION_L_PER_100KM.get(vehicle_type, 25.0)
    vtype_config = VEHICLE_TYPES.get(vehicle_type, {})
    fuel_type = vtype_config.get("fuel_type", "diesel")
    co2_per_liter = CO2_KG_PER_LITER.get(fuel_type, 2.68)
    cost_per_liter = FUEL_COST_INR_PER_LITER.get(fuel_type, 89.62)

    # Total route metrics
    fuel_liters = (distance_km / 100.0) * fuel_l_per_100
    co2_kg = fuel_liters * co2_per_liter
    fuel_cost_inr = fuel_liters * cost_per_liter

    result: Dict[str, Any] = {
        "route_distance_km": round(distance_km, 2),
        "vehicle_type": vehicle_type,
        "fuel_type": fuel_type,
        "fuel_consumption_liters": round(fuel_liters, 2),
        "co2_emissions_kg": round(co2_kg, 2),
        "fuel_cost_inr": round(fuel_cost_inr, 2),
    }

    # Detour delta (cost of life-saving hazard avoidance)
    if primary_distance_km is not None and primary_distance_km > 0:
        detour_extra_km = max(0.0, distance_km - primary_distance_km)
        primary_fuel = (primary_distance_km / 100.0) * fuel_l_per_100
        primary_co2 = primary_fuel * co2_per_liter
        primary_cost = primary_fuel * cost_per_liter

        extra_fuel = fuel_liters - primary_fuel
        extra_co2 = co2_kg - primary_co2
        extra_cost = fuel_cost_inr - primary_cost

        result["primary_route_km"] = round(primary_distance_km, 2)
        result["detour_extra_km"] = round(detour_extra_km, 2)
        result["detour_extra_fuel_liters"] = round(extra_fuel, 2)
        result["detour_extra_co2_kg"] = round(extra_co2, 2)
        result["detour_extra_cost_inr"] = round(extra_cost, 2)
        result["safety_tradeoff"] = (
            f"+{round(detour_extra_km, 1)}km / +{round(extra_fuel, 1)}L {fuel_type} / "
            f"+{round(extra_co2, 1)}kg CO₂ for hazard avoidance"
        )

    return result


def run_swarm(
    destination_name: str,
    destination_lat: float,
    destination_lon: float,
    requested_items: Dict[str, int],
    custom_thresholds: Optional[Dict[str, float]] = None,
    progress_callback: Optional[Callable[[str, str, float], None]] = None,
) -> SwarmState:
    """
    Execute the multi-agent orchestration graph with real-time state handoffs.

    Args:
        destination_name: Human-readable destination name
        destination_lat: Destination latitude
        destination_lon: Destination longitude
        requested_items: Dict of category → quantity
        custom_thresholds: Optional weather threshold overrides
        progress_callback: Optional callback(agent_name, status, progress_pct)

    Returns:
        Final SwarmState with agent logs, LLM thoughts, ML forecasts, and dispatch order
    """
    state = create_initial_state(
        destination_name=destination_name,
        destination_lat=destination_lat,
        destination_lon=destination_lon,
        requested_items=requested_items,
        custom_thresholds=custom_thresholds,
    )

    logger.info(f"Initiating MultiAgentGraph swarm execution for: {destination_name}")
    graph = MultiAgentGraph()
    final_state = graph.invoke(state, progress_callback=progress_callback)
    return final_state


__all__ = [
    "run_swarm",
    "run_telemetry_agent",
    "run_inventory_agent",
    "run_routing_agent",
    "run_dispatch_agent",
    "compute_carbon_metrics",
]
