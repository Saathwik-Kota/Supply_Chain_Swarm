"""
Multi-Scenario Evaluation Benchmark Suite for Supply Chain Swarm.

Executes realistic Indian disaster logistics scenarios, measures P50/P95/P99
latency, inventory fulfillment rates, route optimality ratios, and multi-warehouse
MILP split efficiency.

Usage:
  python -m evaluation.benchmark
"""

import json
import logging
from typing import Any, Dict, List, Optional

from database.db_manager import reset_db, seed_all
from evaluation.metrics import compute_aggregate_benchmarks, evaluate_swarm_run

logger = logging.getLogger(__name__)

BENCHMARK_SCENARIOS: List[Dict[str, Any]] = [
    {
        "scenario_id": "BM-01-CHENNAI-CYCLONE",
        "destination_name": "Chennai Coastal Emergency Zone",
        "destination_lat": 13.0827,
        "destination_lon": 80.2707,
        "requested_items": {"medical_kits": 25, "water_purification": 20, "food_rations": 50, "tents_shelters": 10},
        "custom_thresholds": {"wind_speed_kmh": 9.0, "precipitation_mm_per_hour": 30.0},
    },
    {
        "scenario_id": "BM-02-PUNE-LANDSLIDE",
        "destination_name": "Pune Western Ghats Emergency Zone",
        "destination_lat": 18.5204,
        "destination_lon": 73.8567,
        "requested_items": {"medical_kits": 20, "tents_shelters": 15, "communication_kits": 8},
        "custom_thresholds": None,
    },
    {
        "scenario_id": "BM-03-GUWAHATI-FLOOD",
        "destination_name": "Guwahati Brahmaputra Flood Zone",
        "destination_lat": 26.1445,
        "destination_lon": 91.7362,
        "requested_items": {"water_purification": 30, "food_rations": 60, "medical_kits": 20},
        "custom_thresholds": None,
    },
    {
        "scenario_id": "BM-04-DELHI-SEISMIC",
        "destination_name": "Delhi NCR Seismic Response Zone",
        "destination_lat": 28.6139,
        "destination_lon": 77.2090,
        "requested_items": {"medical_kits": 40, "tents_shelters": 25, "solar_generators": 10, "blankets": 30},
        "custom_thresholds": None,
    },
    {
        "scenario_id": "BM-05-VIZAG-SPLIT-SURGE",
        "destination_name": "Visakhapatnam Mega-Cyclone High-Volume Surge",
        "destination_lat": 17.6868,
        "destination_lon": 83.2185,
        "requested_items": {"medical_kits": 450, "water_purification": 350, "food_rations": 900},
        "custom_thresholds": None,
    },
]


def run_benchmark_suite(
    scenarios: Optional[List[Dict[str, Any]]] = None,
    reset_between_runs: bool = False,
) -> Dict[str, Any]:
    """
    Execute benchmark scenarios against the full MultiAgentGraph swarm and
    return per-scenario metrics alongside aggregate P50/P95/P99 benchmarks.
    """
    from swarm.orchestrator import run_swarm

    target_scenarios = scenarios or BENCHMARK_SCENARIOS
    if reset_between_runs:
        reset_db()
        seed_all()

    scenario_results: List[Dict[str, Any]] = []
    for sc in target_scenarios:
        state = run_swarm(
            destination_name=sc["destination_name"],
            destination_lat=sc["destination_lat"],
            destination_lon=sc["destination_lon"],
            requested_items=sc["requested_items"],
            custom_thresholds=sc.get("custom_thresholds"),
        )
        metrics = state.get("evaluation_metrics") or evaluate_swarm_run(state)
        metrics["scenario_id"] = sc["scenario_id"]
        scenario_results.append(metrics)

    summary = compute_aggregate_benchmarks(scenario_results)
    return {
        "summary": summary,
        "scenarios": scenario_results,
    }


if __name__ == "__main__":
    seed_all()
    report = run_benchmark_suite(reset_between_runs=True)
    print(json.dumps(report, indent=2, default=str))

