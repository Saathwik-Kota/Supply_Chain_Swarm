"""
Unit tests for Change 12 (Evaluation metrics + percentiles),
Change 13 (RAG NDMA/NDRF retrieval + DispatchAgent binding), and
Change 19 (End-to-end Inventory -> Routing -> Dispatch multi-warehouse split flow).
"""

from unittest.mock import patch

from agents.dispatch_agent import DispatchAgent
from agents.inventory_agent import InventoryAgent
from agents.routing_agent import RoutingAgent
from database.db_manager import reset_db, seed_all
from evaluation.metrics import compute_aggregate_benchmarks, evaluate_swarm_run
from rag.retriever import DisasterRAGRetriever, get_rag_retriever
from swarm.state import create_initial_state
from swarm.tools import tool_retrieve_relief_guidelines


def test_rag_retriever_returns_domain_relevant_docs():
    retriever = get_rag_retriever()
    assert isinstance(retriever, DisasterRAGRetriever)
    assert len(retriever.documents) >= 10

    cyclone_res = retriever.retrieve(
        "cyclone storm surge coastal dispatch cold chain insulin",
        disaster_type="cyclone",
        top_k=3,
    )
    assert len(cyclone_res["results"]) == 3
    scores = [d["similarity_score"] for d in cyclone_res["results"]]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] > 0.1

    tool_out = tool_retrieve_relief_guidelines(
        "earthquake urban search and rescue collapse", disaster_type="earthquake", top_k=2
    )
    assert len(tool_out["results"]) == 2
    assert any(
        "seismic" in d["title"].lower()
        or "earthquake" in d["title"].lower()
        or "earthquake" in d["content"].lower()
        for d in tool_out["results"]
    )


def test_evaluate_swarm_run_and_aggregate_benchmarks():
    sample_state = {
        "swarm_status": "completed",
        "destination_name": "Chennai Evaluation Test",
        "requested_items": {"medical_kits": 100, "food_rations": 200},
        "destination_lat": 13.0827,
        "destination_lon": 80.2707,
        "total_latency_ms": 120.5,
        "hazards": [{"hub_name": "Chennai", "hazard_type": "cyclone"}],
        "selected_warehouse": {"warehouse_id": 2, "lat": 12.9716, "lon": 77.5946, "distance_km": 290.0},
        "multi_warehouse_plan": {
            "is_split_fulfillment": False,
            "depot_count": 1,
            "fulfillment_rate_pct": 100.0,
            "total_requested_units": 300,
            "total_allocated_units": 300,
        },
        "route_result": {"distance_km": 345.0, "hazard_avoided": True, "detour_applied": True},
        "split_routes": [{"warehouse_id": 2, "distance_km": 345.0, "hazard_avoided": True}],
        "split_dispatches": [{"dispatch_id": "DSP-1", "total_weight_kg": 450.0}],
        "dispatch_manifest": {"dispatch_id": "DSP-1", "total_weight_kg": 450.0},
        "demand_forecast": {"predictions": {}},
        "route_risk_score": {"risk_score": 0.25},
        "rag_context": [{"doc_id": "SOP-CYC-01"}],
        "errors": [],
    }
    ev = evaluate_swarm_run(sample_state)
    assert ev["fulfillment_rate_pct"] == 100.0
    assert ev["route_optimality_ratio"] >= 1.0
    assert ev["agent_decision_accuracy_pct"] == 100.0

    agg = compute_aggregate_benchmarks([
        {**ev, "total_latency_ms": 100.0},
        {**ev, "total_latency_ms": 200.0},
        {**ev, "total_latency_ms": 400.0},
    ])
    assert agg["total_runs"] == 3
    assert agg["latency_p50_ms"] == 200.0
    assert agg["latency_p95_ms"] >= 200.0
    assert agg["latency_p99_ms"] >= agg["latency_p95_ms"]
    assert agg["pass_rate_pct"] == 100.0


def test_multi_warehouse_split_pipeline_with_rag():
    reset_db()
    seed_all()
    try:
        state = create_initial_state(
            destination_name="Visakhapatnam Mega-Cyclone Split Zone",
            destination_lat=17.6868,
            destination_lon=83.2185,
            requested_items={"medical_kits": 450, "water_purification": 350, "food_rations": 900},
        )
        state["hazards"] = [
            {
                "hub_name": "Bhubaneswar Corridor",
                "lat": 20.2961,
                "lon": 85.8245,
                "hazard_type": "cyclone",
                "severity": "critical",
                "radius_km": 60.0,
                "reason": "Severe cyclonic storm",
                "weather_data": {"wind_speed_kmh": 85.0, "wind_gusts_kmh": 110.0, "precipitation_mm": 65.0},
            }
        ]

        def fake_safe_route(olat, olon, dlat, dlon, hz):
            from geopy.distance import geodesic
            dist = round(geodesic((olat, olon), (dlat, dlon)).km * 1.15, 1)
            return {
                "success": True,
                "hazard_avoided": False,
                "detour_applied": False,
                "detour_waypoints": [],
                "attempts": 1,
                "reasoning_trace": [{"attempt": 1, "status": "ok"}],
                "route": {
                    "origin": [olat, olon],
                    "destination": [dlat, dlon],
                    "distance_km": dist,
                    "duration_min": round(dist * 1.1, 1),
                    "encoded_polyline": "test_poly",
                    "decoded_points": [[olat, olon], [dlat, dlon]],
                },
            }

        with patch("agents.routing_agent.tool_find_safe_route", side_effect=fake_safe_route):
            state = InventoryAgent().run(state)
            assert state["multi_warehouse_plan"]["is_split_fulfillment"] is True
            assert state["multi_warehouse_plan"]["depot_count"] >= 2

            state = RoutingAgent().run(state)
            assert len(state["split_routes"]) == state["multi_warehouse_plan"]["depot_count"]

            state = DispatchAgent().run(state)
            assert state["swarm_status"] == "completed"
            assert len(state["rag_context"]) >= 1
            assert len(state["split_dispatches"]) == state["multi_warehouse_plan"]["depot_count"]
            assert state["carbon_metrics"]["leg_count"] == state["multi_warehouse_plan"]["depot_count"]
    finally:
        reset_db()
        seed_all()
