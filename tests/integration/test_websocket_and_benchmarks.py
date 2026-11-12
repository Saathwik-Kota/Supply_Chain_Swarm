"""
Integration tests for Change 12 (Multi-scenario benchmarks), Change 13 (/api/rag/query),
Change 15 (FastAPI WebSocket /ws/swarm & /ws/events streaming), and
Change 18 & 19 (/api/health/resilience & /api/inventory/optimize-split).
"""

import asyncio
from typing import Any, Dict, List
from unittest.mock import patch

from fastapi import WebSocketDisconnect

from database.db_manager import reset_db, seed_all
from evaluation.benchmark import BENCHMARK_SCENARIOS, run_benchmark_suite
from services.supplier_api import (
    RAGQueryRequest,
    SplitOptimizeRequest,
    api_optimize_multi_warehouse,
    api_query_rag_guidelines,
    execute_swarm_over_websocket,
    resilience_health,
    websocket_event_hub,
)
from swarm.agent_framework import _emit_swarm_event


class FakeWebSocket:
    """In-process async WebSocket double avoiding Python 3.14 AnyIO portal deadlocks."""

    def __init__(self, incoming_json: Dict[str, Any] = None, incoming_texts: List[str] = None):
        self.accepted = False
        self.sent_frames: List[Dict[str, Any]] = []
        self._incoming_json = incoming_json or {}
        self._incoming_texts = list(incoming_texts or [])

    async def accept(self) -> None:
        self.accepted = True

    async def receive_json(self) -> Dict[str, Any]:
        return self._incoming_json

    async def receive_text(self) -> str:
        if self._incoming_texts:
            await asyncio.sleep(0.05)
            return self._incoming_texts.pop(0)
        await asyncio.sleep(1.0)
        raise WebSocketDisconnect(code=1000)

    async def send_json(self, data: Dict[str, Any]) -> None:
        self.sent_frames.append(data)


def _offline_mock_env():
    """Patch weather, disaster, and OSRM network calls for fast offline integration testing."""
    def fake_safe_route(olat, olon, dlat, dlon, hz):
        from geopy.distance import geodesic
        dist = round(geodesic((olat, olon), (dlat, dlon)).km * 1.15, 1)
        return {
            "success": True,
            "hazard_avoided": len(hz) > 0,
            "detour_applied": len(hz) > 0,
            "detour_waypoints": [],
            "attempts": 1,
            "reasoning_trace": [{"attempt": 1, "status": "ok"}],
            "route": {
                "origin": [olat, olon],
                "destination": [dlat, dlon],
                "distance_km": dist,
                "duration_min": round(dist * 1.1, 1),
                "encoded_polyline": "mock_poly",
                "decoded_points": [[olat, olon], [dlat, dlon]],
            },
        }

    fake_scan = {
        "scan_timestamp": "2026-11-11T00:00:00Z",
        "total_hubs": 2,
        "hazardous_hubs": 1,
        "hazards": [
            {
                "hub_name": "Chennai",
                "lat": 13.0827,
                "lon": 80.2707,
                "hazard_type": "cyclone",
                "severity": "high",
                "radius_km": 45.0,
                "reason": "High wind gust alert",
                "weather_data": {"wind_speed_kmh": 70.0, "wind_gusts_kmh": 95.0, "precipitation_mm": 40.0},
            }
        ],
        "hubs": [],
    }
    return fake_safe_route, fake_scan


def test_new_rest_endpoints_resilience_rag_and_split():
    reset_db()
    seed_all()
    try:
        health = resilience_health()
        assert "database" in health
        assert "circuit_breakers" in health
        assert "cache_and_rate_limits" in health

        rag_out = api_query_rag_guidelines(
            RAGQueryRequest(query="flood water purification NDMA guidelines", disaster_type="flood", top_k=2)
        )
        assert len(rag_out["results"]) == 2

        split_out = api_optimize_multi_warehouse(
            SplitOptimizeRequest(
                dest_lat=17.6868,
                dest_lon=83.2185,
                requested_items={"medical_kits": 450, "water_purification": 350, "food_rations": 900},
                max_depots=4,
            )
        )
        assert split_out["is_split_fulfillment"] is True
        assert split_out["fulfillment_rate_pct"] == 100.0
    finally:
        reset_db()
        seed_all()


def test_websocket_swarm_streaming_end_to_end():
    reset_db()
    seed_all()
    fake_safe_route, fake_scan = _offline_mock_env()
    ws = FakeWebSocket(
        incoming_json={
            "destination_name": "Chennai WS Integration Test",
            "destination_lat": 13.0827,
            "destination_lon": 80.2707,
            "requested_items": {"medical_kits": 20, "food_rations": 40},
        }
    )

    try:
        with patch("services.weather_service.WeatherService.scan_all_corridors", return_value=fake_scan), \
             patch("services.disaster_service.DisasterService.get_regional_hazards", return_value=[]), \
             patch("agents.routing_agent.tool_find_safe_route", side_effect=fake_safe_route):
            asyncio.run(execute_swarm_over_websocket(ws))

        assert ws.accepted is True
        events = [f["event"] for f in ws.sent_frames]
        assert "ack" in events
        assert "swarm_started" in events
        assert "agent_started" in events
        assert "agent_completed" in events
        assert "swarm_completed" in events
        assert events[-1] == "swarm_result"

        final_frame = ws.sent_frames[-1]
        assert final_frame["swarm_status"] == "completed"
        assert final_frame["evaluation_metrics"]["fulfillment_rate_pct"] == 100.0
        assert len(final_frame["rag_context"]) >= 1
    finally:
        reset_db()
        seed_all()


def test_websocket_event_hub_broadcasts_and_closes():
    ws = FakeWebSocket(incoming_texts=["close"])

    async def runner():
        hub_task = asyncio.create_task(websocket_event_hub(ws))
        await asyncio.sleep(0.01)
        _emit_swarm_event("agent_started", agent_name="TelemetryAgent", status="running", progress=10.0)
        await hub_task

    asyncio.run(runner())
    events = [f["event"] for f in ws.sent_frames]
    assert events[0] == "connected"
    assert "agent_started" in events


def test_benchmark_suite_with_split_scenario():
    reset_db()
    seed_all()
    fake_safe_route, fake_scan = _offline_mock_env()
    subset = [BENCHMARK_SCENARIOS[0], BENCHMARK_SCENARIOS[2], BENCHMARK_SCENARIOS[4]]

    try:
        with patch("services.weather_service.WeatherService.scan_all_corridors", return_value=fake_scan), \
             patch("services.disaster_service.DisasterService.get_regional_hazards", return_value=[]), \
             patch("agents.routing_agent.tool_find_safe_route", side_effect=fake_safe_route):
            report = run_benchmark_suite(scenarios=subset, reset_between_runs=True)

        assert len(report["scenarios"]) == 3
        agg = report["summary"]
        assert agg["total_runs"] == 3
        assert agg["latency_p50_ms"] > 0
        assert agg["latency_p95_ms"] >= agg["latency_p50_ms"]
        assert agg["latency_p99_ms"] >= agg["latency_p95_ms"]
        assert agg["mean_fulfillment_rate_pct"] == 100.0
        # Verify BM-05-VIZAG-SPLIT-SURGE triggered multi-warehouse split
        assert report["scenarios"][2]["depot_count"] >= 2
    finally:
        reset_db()
        seed_all()
