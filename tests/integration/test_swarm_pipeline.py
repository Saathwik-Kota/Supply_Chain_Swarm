"""
Integration tests for the complete Multi-Agent Swarm Orchestration pipeline.
"""

from swarm.agent_framework import MultiAgentGraph
from swarm.orchestrator import run_swarm
from swarm.state import create_initial_state


def test_full_swarm_pipeline_integration(sample_emergency_request):
    """
    Test end-to-end execution of the 4-agent swarm pipeline.
    Validates state handoffs, ML model evaluations, LLM thoughts, and dispatch generation.
    """
    progress_events = []

    def progress_callback(agent, status, pct):
        progress_events.append((agent, status, pct))

    state = run_swarm(
        destination_name=sample_emergency_request["destination_name"],
        destination_lat=sample_emergency_request["destination_lat"],
        destination_lon=sample_emergency_request["destination_lon"],
        requested_items=sample_emergency_request["requested_items"],
        progress_callback=progress_callback,
    )

    # 1. Swarm Completion Status
    assert state.get("errors") == []
    assert state.get("swarm_status") in ["completed", "partial"]
    assert state.get("total_latency_ms") is not None
    assert state["total_latency_ms"] > 0

    # 2. Telemetry Output
    assert state.get("weather_scan") is not None
    assert isinstance(state.get("hazards"), list)

    # 3. ML Intelligence
    assert state.get("demand_forecast") is not None
    assert "predicted_demands" in state["demand_forecast"]
    assert state.get("route_risk_score") is not None
    assert "risk_level" in state["route_risk_score"]

    # 4. Inventory Allocation
    assert state.get("selected_warehouse") is not None
    assert len(state.get("inventory_checks", [])) > 0

    # 5. Dynamic Routing & Detour
    assert state.get("route_result") is not None
    assert state["route_result"]["distance_km"] > 0

    # 6. Dispatch Execution & Manifest
    assert state.get("dispatch_manifest") is not None
    assert state["dispatch_manifest"]["dispatch_id"].startswith("DSP-")
    assert state["dispatch_manifest"]["destination_name"] == sample_emergency_request["destination_name"]

    # 7. Carbon Impact Accounting
    assert state.get("carbon_metrics") is not None
    assert state["carbon_metrics"]["co2_emissions_kg"] >= 0

    # 8. LLM Reasoning Traces
    llm_reasoning = state.get("llm_reasoning", {})
    assert len(llm_reasoning) >= 3, "Expected recorded LLM thoughts from at least 3 agents"

    # 9. Progress Streaming Callbacks
    assert len(progress_events) >= 4, "Expected progress events for all 4 agents"


def test_multi_agent_graph_abort_on_empty_inventory():
    """Verify state graph conditional edge properly halts on critical failure."""
    graph = MultiAgentGraph()
    state = create_initial_state(
        destination_name="Remote Unreachable Island",
        destination_lat=0.0,
        destination_lon=0.0,
        requested_items={"medical_kits": 10},
    )
    # If selected_warehouse is explicitly None after inventory node, graph handles abort
    state["selected_warehouse"] = None
    next_step = graph.evaluate_inventory_branch(state)
    assert next_step == "abort"
    assert state["swarm_status"] == "failed"

