"""
Quantitative Evaluation Metrics for the Multi-Agent Supply Chain Swarm.

Computes per-mission KPIs and aggregate P50 / P95 / P99 benchmark statistics:
  - Inventory Fulfillment Rate (%)
  - Route Optimality Ratio (Actual Route km / Direct Geodesic km)
  - Strategic Reserve Compliance (%)
  - Agent Decision Accuracy Score (%)
  - Execution Latency Distribution (P50, P95, P99)
"""

import logging
from typing import Any, Dict, List, Mapping

import numpy as np
from geopy.distance import geodesic

logger = logging.getLogger(__name__)


def evaluate_swarm_run(state: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Compute rigorous quantitative evaluation metrics from a completed SwarmState.
    """
    requested = {k: int(v) for k, v in (state.get("requested_items") or {}).items() if int(v) > 0}
    total_requested_units = sum(requested.values())

    # 1. Inventory Fulfillment Rate
    multi_plan = state.get("multi_warehouse_plan")
    deductions = state.get("inventory_deductions") or []

    if multi_plan and isinstance(multi_plan.get("fulfillment_rate_pct"), (int, float)):
        fulfillment_rate = float(multi_plan["fulfillment_rate_pct"])
        fulfilled_units = int(multi_plan.get("total_allocated_units", total_requested_units))
    elif deductions:
        fulfilled_units = sum(int(d.get("deducted", 0)) for d in deductions if d.get("success"))
        fulfillment_rate = (
            round(min(100.0, (fulfilled_units / total_requested_units) * 100.0), 2)
            if total_requested_units > 0
            else 100.0
        )
    else:
        fulfilled_units = 0
        fulfillment_rate = 0.0

    # 2. Route Optimality Ratio (Actual Distance / Ellipsoidal Geodesic Baseline)
    route = state.get("route_result") or {}
    selected = state.get("selected_warehouse") or {}
    actual_dist_km = float(route.get("distance_km") or 0.0)
    geodesic_km = 0.0
    if selected.get("lat") is not None and state.get("destination_lat") is not None:
        geodesic_km = float(
            geodesic(
                (selected["lat"], selected["lon"]),
                (state["destination_lat"], state["destination_lon"]),
            ).km
        )

    if geodesic_km > 0.5 and actual_dist_km > 0:
        route_optimality_ratio = round(actual_dist_km / geodesic_km, 3)
    elif actual_dist_km > 0:
        route_optimality_ratio = 1.0
    else:
        route_optimality_ratio = 0.0

    primary_dist = route.get("primary_route_distance_km")
    detour_overhead_ratio = (
        round(actual_dist_km / float(primary_dist), 3)
        if primary_dist and float(primary_dist) > 0 and route.get("hazard_avoided")
        else 1.0
    )

    # 3. Hazard & Reserve Compliance
    collision_data = route.get("collision_data") or {}
    if route.get("hazard_avoided"):
        # Detour was triggered and applied successfully
        hazard_avoidance_ok = bool(route.get("detour_applied"))
    else:
        hazard_avoidance_ok = not bool(collision_data.get("intersects", False))

    reserve_violations = sum(
        1 for d in deductions
        if not d.get("success") and "Insufficient stock" in str(d.get("error", ""))
    )
    reserve_compliance_pct = 100.0 if reserve_violations == 0 else max(
        0.0, round(100.0 * (1.0 - reserve_violations / max(1, len(deductions))), 2)
    )

    # 4. Agent Decision Accuracy Score (4 balanced criteria x 25 pts)
    score = 0.0
    if selected and fulfillment_rate > 0:
        score += 25.0 * (fulfillment_rate / 100.0)
    vehicle = state.get("vehicle_assignment") or {}
    dispatch = state.get("dispatch_manifest") or {}
    if dispatch.get("dispatch_id") and (
        not vehicle.get("capacity_kg")
        or float(vehicle["capacity_kg"]) >= float(dispatch.get("total_weight_kg") or 0.0)
    ):
        score += 25.0
    if route and hazard_avoidance_ok:
        score += 25.0
    if state.get("demand_forecast") and state.get("route_risk_score"):
        score += 20.0
    if state.get("rag_context"):
        score += 5.0

    # 5. Per-agent latency decomposition
    agent_latencies: Dict[str, float] = {}
    for entry in state.get("agent_logs") or []:
        if entry.get("step") == "complete" and entry.get("latency_ms") is not None:
            agent_latencies[entry.get("agent", "unknown")] = float(entry["latency_ms"])

    return {
        "destination": state.get("destination_name"),
        "swarm_status": state.get("swarm_status"),
        "fulfillment_rate_pct": fulfillment_rate,
        "requested_units": total_requested_units,
        "fulfilled_units": fulfilled_units,
        "is_split_fulfillment": bool(multi_plan and multi_plan.get("is_split_fulfillment")),
        "depot_count": int(multi_plan.get("depot_count", 1)) if multi_plan else (1 if selected else 0),
        "actual_route_km": round(actual_dist_km, 2),
        "geodesic_baseline_km": round(geodesic_km, 2),
        "route_optimality_ratio": route_optimality_ratio,
        "detour_applied": bool(route.get("hazard_avoided")),
        "detour_overhead_ratio": detour_overhead_ratio,
        "hazard_avoidance_verified": hazard_avoidance_ok,
        "reserve_compliance_pct": reserve_compliance_pct,
        "agent_decision_accuracy_pct": round(min(100.0, score), 2),
        "total_latency_ms": float(state.get("total_latency_ms") or 0.0),
        "agent_latencies_ms": agent_latencies,
        "error_count": len(state.get("errors") or []),
    }


def compute_aggregate_benchmarks(run_evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute P50 / P95 / P99 latency and aggregate quality metrics across benchmark runs.
    """
    if not run_evaluations:
        return {
            "total_runs": 0,
            "latency_p50_ms": 0.0,
            "latency_p95_ms": 0.0,
            "latency_p99_ms": 0.0,
            "latency_mean_ms": 0.0,
            "mean_fulfillment_rate_pct": 0.0,
            "mean_route_optimality_ratio": 0.0,
            "mean_decision_accuracy_pct": 0.0,
            "pass_rate_pct": 0.0,
        }

    latencies = [float(r.get("total_latency_ms", 0.0)) for r in run_evaluations]
    fulfillments = [float(r.get("fulfillment_rate_pct", 0.0)) for r in run_evaluations]
    ratios = [float(r.get("route_optimality_ratio", 0.0)) for r in run_evaluations if r.get("route_optimality_ratio", 0) > 0]
    accuracies = [float(r.get("agent_decision_accuracy_pct", 0.0)) for r in run_evaluations]

    detours = sum(1 for r in run_evaluations if r.get("detour_applied"))
    splits = sum(1 for r in run_evaluations if r.get("is_split_fulfillment"))
    passed = sum(
        1 for r in run_evaluations
        if r.get("swarm_status") in ("completed", "partial") and r.get("error_count", 0) == 0
    )
    n = len(run_evaluations)

    return {
        "total_runs": n,
        "passed_runs": passed,
        "pass_rate_pct": round((passed / n) * 100.0, 2),
        "latency_p50_ms": round(float(np.percentile(latencies, 50)), 2),
        "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2),
        "latency_p99_ms": round(float(np.percentile(latencies, 99)), 2),
        "latency_mean_ms": round(float(np.mean(latencies)), 2),
        "mean_fulfillment_rate_pct": round(float(np.mean(fulfillments)), 2),
        "mean_route_optimality_ratio": round(float(np.mean(ratios)), 3) if ratios else 0.0,
        "mean_decision_accuracy_pct": round(float(np.mean(accuracies)), 2),
        "detour_trigger_rate_pct": round((detours / n) * 100.0, 2),
        "split_fulfillment_rate_pct": round((splits / n) * 100.0, 2),
    }

