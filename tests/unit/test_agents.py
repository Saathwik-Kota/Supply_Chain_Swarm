"""
Unit tests for Autonomous Agents (Telemetry, Inventory, Routing, Dispatch).
"""

from agents.dispatch_agent import DispatchAgent
from agents.inventory_agent import InventoryAgent
from agents.routing_agent import RoutingAgent
from agents.telemetry_agent import TelemetryAgent


def test_telemetry_agent_run(fresh_swarm_state):
    agent = TelemetryAgent()
    state = agent.run(fresh_swarm_state)
    assert state["errors"] == []
    assert state["weather_scan"] is not None
    assert isinstance(state["hazards"], list)
    assert len(state["agent_logs"]) > 0
    assert "telemetry" in state["llm_reasoning"]
    assert len(state["llm_reasoning"]["telemetry"]) >= 2


def test_inventory_agent_run(fresh_swarm_state):
    # Pre-populate dummy hazard
    fresh_swarm_state["hazards"] = [{"hazard_type": "cyclone", "severity": "high", "lat": 13.0, "lon": 80.0}]

    agent = InventoryAgent()
    state = agent.run(fresh_swarm_state)

    assert state["demand_forecast"] is not None
    assert "predicted_demands" in state["demand_forecast"]
    assert state["selected_warehouse"] is not None
    assert len(state["inventory_checks"]) > 0
    assert "inventory" in state["llm_reasoning"]


def test_routing_agent_run(fresh_swarm_state):
    # Pre-populate selected warehouse
    fresh_swarm_state["selected_warehouse"] = {
        "warehouse_id": 1,
        "warehouse_name": "CONCOR CFS Tondiarpet",
        "city": "Chennai",
        "lat": 13.1256,
        "lon": 80.2925,
        "distance_km": 8.0,
    }

    agent = RoutingAgent()
    state = agent.run(fresh_swarm_state)

    assert state["route_result"] is not None
    assert "distance_km" in state["route_result"]
    assert state["route_risk_score"] is not None
    assert "routing" in state["llm_reasoning"]


def test_dispatch_agent_run(fresh_swarm_state):
    # Pre-populate selected warehouse and route
    fresh_swarm_state["selected_warehouse"] = {
        "warehouse_id": 1,
        "warehouse_name": "CONCOR CFS Tondiarpet",
        "city": "Chennai",
        "lat": 13.1256,
        "lon": 80.2925,
        "distance_km": 8.0,
        "available_items": {
            "medical_kits": {"available": True, "available_above_reserve": 50},
            "water_purification": {"available": True, "available_above_reserve": 50},
            "food_rations": {"available": True, "available_above_reserve": 100},
        },
    }
    fresh_swarm_state["route_result"] = {
        "distance_km": 8.5,
        "duration_min": 15.0,
        "hazard_avoided": False,
    }

    agent = DispatchAgent()
    state = agent.run(fresh_swarm_state)

    assert state["dispatch_manifest"] is not None
    assert state["dispatch_manifest"]["dispatch_id"].startswith("DSP-")
    assert state["swarm_status"] == "completed"
    assert "dispatch" in state["llm_reasoning"]

