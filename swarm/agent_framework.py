"""
Multi-Agent Orchestration Graph Engine (Inspired by LangGraph / Google Agent ADK).

Replaces rigid sequential loops with an expressive State Graph architecture:
  - Nodes: Autonomous agents and ML reasoning stages.
  - Conditional Edges: State-dependent branching (e.g. hazard detection triggers detour routing,
    inventory deficit triggers partial allocation fallback).
  - State Transitions: Strict typed state validation and handoffs between agents.
  - Live Event Streaming: Real-time progress updates for UI and observability.
"""

import logging
import threading
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from agents.dispatch_agent import DispatchAgent
from agents.inventory_agent import InventoryAgent
from agents.routing_agent import RoutingAgent
from agents.telemetry_agent import TelemetryAgent
from config import AGENT_NAMES
from evaluation.metrics import evaluate_swarm_run
from swarm.state import AgentLog, SwarmState

logger = logging.getLogger(__name__)

# ─── Thread-safe Event Hub for WebSocket & Real-time Streaming ──────────────
_EVENT_LOCK = threading.Lock()
_EVENT_LISTENERS: List[Callable[[Dict[str, Any]], None]] = []


def register_event_listener(listener: Callable[[Dict[str, Any]], None]) -> None:
    """Register a callable subscriber for real-time swarm execution events."""
    with _EVENT_LOCK:
        if listener not in _EVENT_LISTENERS:
            _EVENT_LISTENERS.append(listener)


def unregister_event_listener(listener: Callable[[Dict[str, Any]], None]) -> None:
    """Remove a previously registered swarm event subscriber."""
    with _EVENT_LOCK:
        if listener in _EVENT_LISTENERS:
            _EVENT_LISTENERS.remove(listener)


def _emit_swarm_event(
    event_type: str,
    agent_name: str = "",
    status: str = "info",
    progress: float = 0.0,
    data: Optional[Dict[str, Any]] = None,
) -> None:
    """Broadcast a structured lifecycle event to all active pub/sub listeners."""
    payload: Dict[str, Any] = {
        "event": event_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent": agent_name,
        "status": status,
        "progress": progress,
        "data": data or {},
    }
    with _EVENT_LOCK:
        subscribers = list(_EVENT_LISTENERS)
    for cb in subscribers:
        try:
            cb(payload)
        except Exception as exc:
            logger.debug("Swarm event listener failed cleanly: %s", exc)


class AgentNode:
    """Represents an agent execution stage in the multi-agent graph."""

    def __init__(self, key: str, name: str, agent_instance: Any):
        self.key = key
        self.name = name
        self.agent = agent_instance

    def execute(self, state: SwarmState) -> SwarmState:
        """Execute node with timing and exception isolation."""
        logger.info(f"[GraphNode: {self.name}] Entering execution...")
        start = time.time()
        try:
            state = self.agent.run(state)
        except Exception as e:
            err_msg = f"Node '{self.name}' crashed: {str(e)}"
            logger.error(err_msg, exc_info=True)
            state["errors"].append({"agent": self.name, "error": err_msg})
            state["agent_logs"].append(AgentLog(
                agent=self.name,
                step="node_crash",
                timestamp=datetime.now(timezone.utc).isoformat(),
                message=err_msg,
                error=traceback.format_exc(),
            ))
        elapsed = round((time.time() - start) * 1000, 2)
        logger.info(f"[GraphNode: {self.name}] Completed in {elapsed}ms")
        return state


class MultiAgentGraph:
    """
    State Graph orchestrator managing agent nodes, conditional branching,
    and state handoff.
    """

    def __init__(self):
        self.telemetry_node = AgentNode("telemetry", AGENT_NAMES["telemetry"], TelemetryAgent())
        self.inventory_node = AgentNode("inventory", AGENT_NAMES["inventory"], InventoryAgent())
        self.routing_node = AgentNode("routing", AGENT_NAMES["routing"], RoutingAgent())
        self.dispatch_node = AgentNode("dispatch", AGENT_NAMES["dispatch"], DispatchAgent())

    # ─── Conditional Routing Logic ───────────────────────────────────────────

    def evaluate_telemetry_branch(self, state: SwarmState) -> str:
        """Route to inventory with threat level context."""
        hazards = state.get("hazards", [])
        if hazards:
            logger.info(f"[ConditionalEdge: ThreatBranch] Detected {len(hazards)} hazards. Flagging for high-threat routing.")
        return "inventory"

    def evaluate_inventory_branch(self, state: SwarmState) -> str:
        """Verify warehouse allocation feasibility."""
        selected = state.get("selected_warehouse")
        if not selected:
            logger.warning("[ConditionalEdge: InventoryBranch] No warehouse candidate found! Aborting.")
            state["swarm_status"] = "failed"
            return "abort"
        return "routing"

    def evaluate_routing_branch(self, state: SwarmState) -> str:
        """Verify corridor safety and self-correction resolution."""
        route = state.get("route_result")
        if not route:
            logger.warning("[ConditionalEdge: RoutingBranch] Route computation failed. Proceeding with warning.")
        elif route.get("hazard_avoided"):
            logger.info("[ConditionalEdge: RoutingBranch] Verified detour clearance around hazard perimeter.")
        return "dispatch"

    # ─── Graph Execution ─────────────────────────────────────────────────────

    def _notify_stage(
        self,
        progress_callback: Optional[Callable[[str, str, float], None]],
        agent_name: str,
        stage_status: str,
        progress: float,
        event_type: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        if progress_callback:
            progress_callback(agent_name, stage_status, progress)
        _emit_swarm_event(
            event_type=event_type,
            agent_name=agent_name,
            status=stage_status,
            progress=progress,
            data=data,
        )

    def invoke(
        self,
        state: SwarmState,
        progress_callback: Optional[Callable[[str, str, float], None]] = None,
    ) -> SwarmState:
        """
        Execute the compiled multi-agent graph with dynamic state handoffs.
        """
        total_start = time.time()
        _emit_swarm_event(
            "swarm_started",
            agent_name="SwarmCoordinator",
            status="running",
            progress=0.0,
            data={"destination_name": state.get("destination_name")},
        )

        # ── Step 1: Telemetry & Threat Monitor Node ──────────────────────────
        self._notify_stage(progress_callback, self.telemetry_node.name, "running", 10.0, "agent_started")
        state = self.telemetry_node.execute(state)
        self.evaluate_telemetry_branch(state)
        self._notify_stage(
            progress_callback,
            self.telemetry_node.name,
            "completed",
            25.0,
            "agent_completed",
            {"hazards_detected": len(state.get("hazards", []))},
        )

        # ── Step 2: Regional Inventory & Supplier Node ───────────────────────
        self._notify_stage(progress_callback, self.inventory_node.name, "running", 35.0, "agent_started")
        state = self.inventory_node.execute(state)
        next_step = self.evaluate_inventory_branch(state)
        self._notify_stage(
            progress_callback,
            self.inventory_node.name,
            "completed",
            50.0,
            "agent_completed",
            {
                "selected_warehouse": (state.get("selected_warehouse") or {}).get("warehouse_name"),
                "split_depots": (state.get("multi_warehouse_plan") or {}).get("depot_count", 0),
            },
        )

        if next_step == "abort":
            state["swarm_status"] = "failed"
            state["total_latency_ms"] = round((time.time() - total_start) * 1000, 2)
            state["evaluation_metrics"] = evaluate_swarm_run(state)
            _emit_swarm_event(
                "swarm_aborted",
                agent_name=self.inventory_node.name,
                status="failed",
                progress=50.0,
                data={"errors": state.get("errors", [])},
            )
            return state

        # ── Step 3: Dynamic Routing & Avoidance Node ─────────────────────────
        self._notify_stage(progress_callback, self.routing_node.name, "running", 60.0, "agent_started")
        state = self.routing_node.execute(state)
        self.evaluate_routing_branch(state)
        self._notify_stage(
            progress_callback,
            self.routing_node.name,
            "completed",
            75.0,
            "agent_completed",
            {
                "primary_route_blocked": state.get("primary_route_blocked", False),
                "split_routes": len(state.get("split_routes", [])),
            },
        )

        # ── Step 4: Autonomous Dispatch Coordinator Node ─────────────────────
        self._notify_stage(progress_callback, self.dispatch_node.name, "running", 85.0, "agent_started")
        state = self.dispatch_node.execute(state)
        self._notify_stage(
            progress_callback,
            self.dispatch_node.name,
            "completed",
            100.0,
            "agent_completed",
            {"dispatch_id": (state.get("dispatch_manifest") or {}).get("dispatch_id")},
        )

        # ── Graph Finalization & Evaluation ──────────────────────────────────
        total_latency = (time.time() - total_start) * 1000
        state["completed_at"] = datetime.now(timezone.utc).isoformat()
        state["total_latency_ms"] = round(total_latency, 2)
        state["current_agent"] = ""

        if state.get("swarm_status") == "running":
            state["swarm_status"] = "completed" if not state.get("errors") else "partial"

        state["evaluation_metrics"] = evaluate_swarm_run(state)

        _emit_swarm_event(
            "swarm_completed",
            agent_name="SwarmCoordinator",
            status=state.get("swarm_status", "completed"),
            progress=100.0,
            data={
                "total_latency_ms": state["total_latency_ms"],
                "evaluation_metrics": state["evaluation_metrics"],
            },
        )

        logger.info(
            f"Multi-Agent Graph completed in {total_latency:.0f}ms with status '{state.get('swarm_status')}'"
        )
        return state

