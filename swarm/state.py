"""
State schema for the multi-agent emergency response swarm.

Defines the TypedDict state that flows through the agent orchestration
graph. Each agent reads from and writes to specific state keys.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypedDict


class HazardInfo(TypedDict, total=False):
    """A detected hazard zone with assessment details."""
    hub_name: str
    lat: float
    lon: float
    hazard_type: str
    severity: str
    weather_data: Dict[str, Any]
    assessment: Dict[str, Any]
    radius_km: float
    source: str
    reason: str
    hazard_zone_id: Optional[int]


class InventoryCheck(TypedDict, total=False):
    """Result of an inventory availability check."""
    warehouse_id: int
    warehouse_name: str
    city: str
    lat: float
    lon: float
    distance_km: float
    available_items: Dict[str, Dict[str, Any]]
    total_weight_kg: float
    can_fulfill: bool
    full_inventory: List[Dict[str, Any]]
    available_vehicles: List[Dict[str, Any]]


class RouteResult(TypedDict, total=False):
    """A computed route with hazard avoidance info."""
    warehouse_id: int
    warehouse_name: str
    origin: Any
    destination: Any
    distance_km: float
    duration_min: float
    encoded_polyline: str
    decoded_points: List[tuple]
    hazard_avoided: bool
    detour_applied: bool
    detour_waypoints: List[tuple]
    primary_route_distance_km: Optional[float]
    distance_increase_km: Optional[float]
    collision_data: Optional[Dict[str, Any]]
    attempts: int


class DispatchManifest(TypedDict, total=False):
    """A completed dispatch order manifest."""
    dispatch_id: str
    source_warehouse_id: int
    source_warehouse_name: str
    vehicle_id: Optional[str]
    vehicle_type: Optional[str]
    destination_name: str
    destination_lat: float
    destination_lon: float
    items: Dict[str, int]
    total_weight_kg: float
    route_distance_km: Optional[float]
    route_duration_min: Optional[float]
    eta_minutes: Optional[float]
    hazard_avoided: bool
    detour_reason: Optional[str]
    status: str
    carbon_metrics: Optional[Dict[str, Any]]


class AgentLog(TypedDict, total=False):
    """A single agent's execution log entry."""
    agent: str
    step: str
    timestamp: str
    message: str
    data: Optional[Dict[str, Any]]
    latency_ms: Optional[float]
    error: Optional[str]


class SwarmState(TypedDict, total=False):
    """
    Complete state flowing through the multi-agent orchestration graph.

    Agent write responsibilities:
      - Telemetry Agent → hazards, weather_scan, telemetry_logs
      - Inventory Agent → inventory_checks, selected_warehouse
      - Routing Agent → route_result, primary_route_blocked
      - Dispatch Agent → dispatch_manifest, inventory_deductions
    """
    # ─── Input Parameters ────────────────────────────────────────────────
    destination_name: str
    destination_lat: float
    destination_lon: float
    requested_items: Dict[str, int]  # category → quantity
    custom_thresholds: Optional[Dict[str, float]]  # override weather thresholds

    # ─── Telemetry Agent Output ──────────────────────────────────────────
    weather_scan: Optional[Dict[str, Any]]
    hazards: List[HazardInfo]
    hazard_zones_db: List[Dict[str, Any]]  # Hazard zones written to DB
    telemetry_logs: List[Dict[str, Any]]

    # ─── ML Intelligence Outputs ─────────────────────────────────────────
    demand_forecast: Optional[Dict[str, Any]]  # Time-series ML demand predictions
    route_risk_score: Optional[Dict[str, Any]]  # ML route risk & anomaly detection

    # ─── Inventory Agent Output ──────────────────────────────────────────
    inventory_checks: List[InventoryCheck]
    selected_warehouse: Optional[InventoryCheck]
    multi_warehouse_plan: Optional[Dict[str, Any]]

    # ─── Routing Agent Output ────────────────────────────────────────────
    route_result: Optional[RouteResult]
    split_routes: List[Dict[str, Any]]
    primary_route_blocked: bool
    routing_reasoning_trace: List[Dict[str, Any]]

    # ─── Dispatch Agent Output ───────────────────────────────────────────
    dispatch_manifest: Optional[DispatchManifest]
    split_dispatches: List[DispatchManifest]
    inventory_deductions: List[Dict[str, Any]]
    vehicle_assignment: Optional[Dict[str, Any]]
    carbon_metrics: Optional[Dict[str, Any]]
    rag_context: List[Dict[str, Any]]

    # ─── Shared Execution Metadata & Evaluation Metrics ──────────────────
    evaluation_metrics: Optional[Dict[str, Any]]
    agent_logs: List[AgentLog]
    llm_reasoning: Dict[str, List[Dict[str, Any]]]  # Per-agent LLM chain-of-thought & tool decisions
    current_agent: str
    errors: List[Dict[str, Any]]
    swarm_status: str  # "running", "completed", "failed", "partial"
    started_at: str
    completed_at: Optional[str]
    total_latency_ms: Optional[float]


def create_initial_state(
    destination_name: str,
    destination_lat: float,
    destination_lon: float,
    requested_items: Dict[str, int],
    custom_thresholds: Optional[Dict[str, float]] = None,
) -> SwarmState:
    """Create a fresh swarm state with input parameters."""
    return SwarmState(
        destination_name=destination_name,
        destination_lat=destination_lat,
        destination_lon=destination_lon,
        requested_items=requested_items,
        custom_thresholds=custom_thresholds,
        weather_scan=None,
        hazards=[],
        hazard_zones_db=[],
        telemetry_logs=[],
        demand_forecast=None,
        route_risk_score=None,
        inventory_checks=[],
        selected_warehouse=None,
        multi_warehouse_plan=None,
        route_result=None,
        split_routes=[],
        primary_route_blocked=False,
        routing_reasoning_trace=[],
        dispatch_manifest=None,
        split_dispatches=[],
        inventory_deductions=[],
        vehicle_assignment=None,
        carbon_metrics=None,
        rag_context=[],
        evaluation_metrics=None,
        agent_logs=[],
        llm_reasoning={},
        current_agent="",
        errors=[],
        swarm_status="running",
        started_at=datetime.now(timezone.utc).isoformat(),
        completed_at=None,
        total_latency_ms=None,
    )

