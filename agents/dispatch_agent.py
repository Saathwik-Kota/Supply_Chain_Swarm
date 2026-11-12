"""
Agent 4: Autonomous Dispatch & Execution Coordinator Agent

Executes the finalized emergency relief dispatch order:
  1. Atomically deducts inventory from the selected origin warehouse.
  2. Allocates optimal transport fleet vehicle based on total payload weight.
  3. Computes environmental carbon footprint, fuel burn, and detour safety cost trade-offs.
  4. Commits official dispatch order and manifest into the persistent database.
"""

import json
import logging
from typing import Any, Callable, Dict, List, Optional, cast

from agents.base_agent import BaseAgent
from config import AGENT_NAMES
from llm.prompts import DISPATCH_AGENT_SYSTEM_PROMPT
from swarm.state import DispatchManifest, SwarmState
from swarm.tools import (
    tool_assign_vehicle,
    tool_compute_carbon_and_costs,
    tool_create_dispatch_order,
    tool_deduct_inventory,
    tool_get_available_fleet,
    tool_retrieve_relief_guidelines,
)

logger = logging.getLogger(__name__)


class DispatchAgent(BaseAgent):
    """
    Autonomous dispatch and fleet execution coordinator powered by RAG SOP retrieval,
    multi-depot split execution, and LLM decision making.
    """

    def __init__(self):
        tools: Dict[str, Callable[..., Any]] = {
            "tool_deduct_inventory": tool_deduct_inventory,
            "tool_assign_vehicle": tool_assign_vehicle,
            "tool_compute_carbon_and_costs": tool_compute_carbon_and_costs,
            "tool_create_dispatch_order": tool_create_dispatch_order,
            "tool_get_available_fleet": tool_get_available_fleet,
            "tool_retrieve_relief_guidelines": tool_retrieve_relief_guidelines,
        }
        super().__init__(
            name=AGENT_NAMES["dispatch"],
            role_key="dispatch",
            system_prompt=DISPATCH_AGENT_SYSTEM_PROMPT,
            tool_names=list(tools.keys()),
            tool_registry=tools,
            max_react_steps=3,
        )

    def _retrieve_operational_sops(self, state: SwarmState) -> List[Dict[str, Any]]:
        """Retrieve relevant NDMA/NDRF/cold-chain SOP guidelines via RAG."""
        hazards = state.get("hazards", [])
        hazard_type = hazards[0].get("hazard_type", "cyclone") if hazards else None
        req_cats = ", ".join((state.get("requested_items") or {}).keys()) or "relief supplies"
        query = (
            f"{hazard_type or 'disaster'} emergency logistics dispatch SOP for "
            f"{state.get('destination_name', 'affected zone')} handling {req_cats} "
            f"cold chain and hazard detour safety"
        )
        rag_res = tool_retrieve_relief_guidelines(query=query, disaster_type=hazard_type, top_k=3)
        docs = cast(List[Dict[str, Any]], rag_res.get("results", []))
        state["rag_context"] = docs
        top_titles = ", ".join(f"{d['title']} ({d['similarity_score']:.2f})" for d in docs)
        self.log(
            state,
            "rag_retrieval",
            f"📚 RAG Retriever fetched {len(docs)} NDMA/NDRF protocol doc(s) in {rag_res.get('latency_ms', 0)}ms: {top_titles}",
            data=rag_res,
        )
        return docs

    def _execute_depot_leg(
        self,
        state: SwarmState,
        wh_id: int,
        wh_name: str,
        items_to_deduct: Dict[str, int],
        leg_route: Optional[Dict[str, Any]],
        reasoning_summary: str,
    ) -> Dict[str, Any]:
        """Execute ACID deduction, vehicle assignment, carbon calc, and dispatch order for one depot leg."""
        deduction_res = tool_deduct_inventory(wh_id, items_to_deduct)
        deductions = deduction_res.get("deductions", [])
        total_weight = sum(
            d.get("weight_deducted_kg", 0) for d in deductions if d.get("success")
        )

        fleet = tool_get_available_fleet(wh_id, total_weight)
        vehicle_assigned = None
        if fleet:
            best_vehicle = fleet[0]
            assign_res = tool_assign_vehicle(best_vehicle["vehicle_id"])
            if assign_res.get("success"):
                vehicle_assigned = assign_res
                self.log(
                    state,
                    "vehicle_assigned",
                    f"✅ Assigned vehicle {best_vehicle['vehicle_id']} ({best_vehicle['display_name']}) "
                    f"at {wh_name}, payload cap: {best_vehicle['capacity_kg']}kg",
                    data=assign_res,
                )

        v_type = vehicle_assigned.get("type", "medium_truck") if vehicle_assigned else "medium_truck"
        route_dist = leg_route.get("distance_km") if leg_route else 0.0
        primary_dist = leg_route.get("primary_route_distance_km") if leg_route else None

        carbon = None
        if route_dist and route_dist > 0:
            carbon = tool_compute_carbon_and_costs(
                distance_km=route_dist,
                vehicle_type=v_type,
                primary_distance_km=primary_dist,
            )

        detour_reason = None
        if leg_route and leg_route.get("hazard_avoided"):
            detour_reason = (
                f"Primary corridor obstructed by {len(state.get('hazards', []))} hazard zone(s). "
                f"Detour overhead: +{leg_route.get('distance_increase_km', 0)}km."
            )

        dispatch_res = tool_create_dispatch_order(
            source_warehouse_id=wh_id,
            destination_name=state["destination_name"],
            destination_lat=state["destination_lat"],
            destination_lon=state["destination_lon"],
            items_manifest=items_to_deduct,
            vehicle_id=vehicle_assigned["vehicle_id"] if vehicle_assigned else None,
            route_geometry=leg_route.get("encoded_polyline") if leg_route else None,
            route_distance_km=leg_route.get("distance_km") if leg_route else None,
            route_duration_min=leg_route.get("duration_min") if leg_route else None,
            eta_minutes=leg_route.get("duration_min") if leg_route else None,
            hazard_avoided=leg_route.get("hazard_avoided", False) if leg_route else False,
            detour_reason=detour_reason,
            agent_reasoning_log=reasoning_summary,
        )

        manifest: DispatchManifest = {
            "dispatch_id": dispatch_res["dispatch_id"],
            "source_warehouse_id": dispatch_res["source_warehouse_id"],
            "source_warehouse_name": wh_name,
            "vehicle_id": dispatch_res.get("vehicle_id"),
            "vehicle_type": vehicle_assigned.get("type") if vehicle_assigned else None,
            "destination_name": dispatch_res["destination"],
            "destination_lat": state["destination_lat"],
            "destination_lon": state["destination_lon"],
            "items": dispatch_res["items_manifest"],
            "total_weight_kg": dispatch_res["total_weight_kg"],
            "route_distance_km": dispatch_res.get("route_distance_km"),
            "route_duration_min": dispatch_res.get("route_duration_min"),
            "eta_minutes": dispatch_res.get("eta_minutes"),
            "hazard_avoided": dispatch_res.get("hazard_avoided", False),
            "detour_reason": detour_reason,
            "status": dispatch_res["status"],
            "carbon_metrics": carbon,
        }
        return {
            "deductions": deductions,
            "vehicle_assigned": vehicle_assigned,
            "carbon": carbon,
            "dispatch_res": dispatch_res,
            "manifest": manifest,
        }

    def run(self, state: SwarmState) -> SwarmState:
        state["current_agent"] = self.name
        self.log(state, "start", f"Activated {self.name} with {len(self.tool_names)} registered tools.")

        selected = state.get("selected_warehouse")
        route = state.get("route_result")

        if not selected:
            err = "No warehouse selected — dispatch execution aborted."
            self.log(state, "no_warehouse", f"❌ {err}", error=err)
            state["errors"].append({"agent": self.name, "error": err})
            state["swarm_status"] = "failed"
            return state

        # 0. Retrieve NDMA/NDRF Operational Guidelines via RAG
        rag_docs = self._retrieve_operational_sops(state)

        # Build leg plan (multi-warehouse split vs backward-compatible single depot)
        multi_plan = state.get("multi_warehouse_plan") or {}
        allocations = multi_plan.get("allocations", [])
        legs: List[Dict[str, Any]] = []
        if multi_plan.get("is_split_fulfillment") and len(allocations) > 1:
            legs = [
                {
                    "warehouse_id": int(alloc["warehouse_id"]),
                    "warehouse_name": str(alloc.get("warehouse_name", f"Depot-{alloc['warehouse_id']}")),
                    "items": {str(k): int(v) for k, v in dict(alloc.get("allocated_items", {})).items()},
                }
                for alloc in allocations
                if alloc.get("allocated_items")
            ]
        else:
            items_to_deduct: Dict[str, int] = {}
            for cat, qty in state.get("requested_items", {}).items():
                item_check = selected.get("available_items", {}).get(cat, {})
                if item_check.get("available", False):
                    items_to_deduct[cat] = int(qty)
                else:
                    avail = int(item_check.get("available_above_reserve", 0))
                    if avail > 0:
                        items_to_deduct[cat] = min(int(qty), avail)
            legs = [
                {
                    "warehouse_id": int(selected["warehouse_id"]),
                    "warehouse_name": str(selected["warehouse_name"]),
                    "items": items_to_deduct,
                }
            ]

        split_routes = state.get("split_routes") or []
        reasoning_summary = json.dumps(
            [{"agent": entry["agent"], "step": entry["step"], "message": entry["message"]} for entry in state["agent_logs"]],
            indent=2,
        )

        all_deductions: List[Dict[str, Any]] = []
        split_dispatches: List[DispatchManifest] = []
        primary_vehicle = None
        carbon_legs: List[Dict[str, Any]] = []

        for idx, leg in enumerate(legs):
            wh_id = int(leg["warehouse_id"])
            wh_name = str(leg["warehouse_name"])
            leg_items: Dict[str, int] = dict(leg["items"])
            leg_route: Optional[Dict[str, Any]] = next(
                (r for r in split_routes if r.get("warehouse_id") == wh_id), None
            )
            if leg_route is None and wh_id == selected["warehouse_id"] and route is not None:
                leg_route = dict(route)

            leg_out = self._execute_depot_leg(
                state=state,
                wh_id=wh_id,
                wh_name=wh_name,
                items_to_deduct=leg_items,
                leg_route=leg_route,
                reasoning_summary=reasoning_summary,
            )
            all_deductions.extend(leg_out["deductions"])
            if idx == 0 and leg_out["vehicle_assigned"]:
                primary_vehicle = leg_out["vehicle_assigned"]
            if leg_out["carbon"]:
                carbon_legs.append(leg_out["carbon"])
            split_dispatches.append(leg_out["manifest"])

            self.log(
                state,
                "inventory_deducted" if idx == 0 else "split_leg_dispatched",
                f"✅ {'Inventory deducted & dispatch' if idx == 0 else 'Split depot leg dispatch'} "
                f"{leg_out['dispatch_res']['dispatch_id']} committed at {wh_name} "
                f"for {len(leg_items)} categories ({leg_out['dispatch_res']['total_weight_kg']}kg).",
                data=leg_out["dispatch_res"],
            )

        state["inventory_deductions"] = all_deductions
        if primary_vehicle:
            state["vehicle_assignment"] = primary_vehicle

        if carbon_legs:
            if len(carbon_legs) == 1:
                agg_carbon = carbon_legs[0]
            else:
                agg_carbon = dict(carbon_legs[0])
                for metric_key in (
                    "route_distance_km",
                    "fuel_consumption_liters",
                    "co2_emissions_kg",
                    "fuel_cost_inr",
                    "detour_extra_km",
                    "detour_extra_fuel_liters",
                    "detour_extra_co2_kg",
                    "detour_extra_cost_inr",
                ):
                    if any(metric_key in c for c in carbon_legs):
                        agg_carbon[metric_key] = round(
                            sum(float(c.get(metric_key, 0.0)) for c in carbon_legs), 2
                        )
                agg_carbon["leg_count"] = len(carbon_legs)
            state["carbon_metrics"] = agg_carbon
            self.log(
                state,
                "carbon_impact",
                f"🌿 Carbon Analytics: {agg_carbon['fuel_consumption_liters']}L {agg_carbon['fuel_type']} → "
                f"{agg_carbon['co2_emissions_kg']}kg CO₂ | Fuel cost: ₹{agg_carbon['fuel_cost_inr']:,}.",
                data=agg_carbon,
            )

        state["split_dispatches"] = split_dispatches
        state["dispatch_manifest"] = split_dispatches[0]
        state["swarm_status"] = "completed"

        primary_manifest = split_dispatches[0]
        total_dispatched_weight = round(sum(float(m.get("total_weight_kg", 0)) for m in split_dispatches), 2)
        self.log(
            state,
            "dispatch_created",
            f"🚀 OFFICIAL DISPATCH ORDER ISSUED: {primary_manifest['dispatch_id']} "
            f"({len(split_dispatches)} depot leg(s)). Combined Weight: {total_dispatched_weight}kg, "
            f"Primary ETA: {primary_manifest.get('eta_minutes')} min.",
            data={"primary_dispatch": primary_manifest, "split_dispatches": split_dispatches},
        )

        # Record LLM reasoning
        if "llm_reasoning" not in state or state["llm_reasoning"] is None:
            state["llm_reasoning"] = {}

        co2_val = (state.get("carbon_metrics") or {}).get("co2_emissions_kg", 0)
        state["llm_reasoning"]["dispatch"] = [{
            "step": 1,
            "thought": (
                f"Retrieved {len(rag_docs)} NDMA/NDRF SOP guidelines via RAG and committed "
                f"{len(split_dispatches)} ACID relief dispatch order(s) starting with {primary_manifest['dispatch_id']}. "
                f"Assigned transport vehicle {primary_vehicle.get('vehicle_id') if primary_vehicle else 'N/A'}, "
                f"calculated total carbon footprint ({co2_val} kg CO₂)."
            ),
            "tool_calls": [
                "tool_retrieve_relief_guidelines",
                "tool_deduct_inventory",
                "tool_assign_vehicle",
                "tool_compute_carbon_and_costs",
                "tool_create_dispatch_order",
            ],
            "provider": self.llm_client.provider,
        }]

        return state


# Singleton and convenience executor
_GLOBAL_DISPATCH_AGENT = DispatchAgent()

def run_dispatch_agent(state: SwarmState) -> SwarmState:
    """Entry point for Dispatch Agent execution."""
    return _GLOBAL_DISPATCH_AGENT.run(state)

__all__ = ["DispatchAgent", "run_dispatch_agent"]
