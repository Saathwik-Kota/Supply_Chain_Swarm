"""
Agent 1: Telemetry & Threat Monitor Agent

Continuously polls live weather telemetry (Open-Meteo API) across active Indian logistics corridors.
Ingests live USGS seismic and NASA EONET feeds. Assesses hazards against configurable thresholds,
evaluates meteorological risk, and registers active hazard zones into the persistent database.
"""

import logging
from typing import Any, Callable, Dict, List

from agents.base_agent import BaseAgent
from config import AGENT_NAMES, HAZARD_RADIUS_KM
from llm.prompts import TELEMETRY_AGENT_SYSTEM_PROMPT
from swarm.state import HazardInfo, SwarmState
from swarm.tools import (
    tool_assess_route_risk,
    tool_get_active_hazard_zones_from_db,
    tool_register_hazard_zone,
    tool_scan_custom_location,
    tool_scan_live_disaster_feeds,
    tool_scan_weather_corridors,
)

logger = logging.getLogger(__name__)


class TelemetryAgent(BaseAgent):
    """
    Telemetry and environmental threat monitor agent powered by LLM decision making.
    """

    def __init__(self):
        tools: Dict[str, Callable[..., Any]] = {
            "tool_scan_weather_corridors": tool_scan_weather_corridors,
            "tool_scan_custom_location": tool_scan_custom_location,
            "tool_scan_live_disaster_feeds": tool_scan_live_disaster_feeds,
            "tool_register_hazard_zone": tool_register_hazard_zone,
            "tool_get_active_hazard_zones_from_db": tool_get_active_hazard_zones_from_db,
            "tool_assess_route_risk": tool_assess_route_risk,
        }
        super().__init__(
            name=AGENT_NAMES["telemetry"],
            role_key="telemetry",
            system_prompt=TELEMETRY_AGENT_SYSTEM_PROMPT,
            tool_names=list(tools.keys()),
            tool_registry=tools,
            max_react_steps=3,
        )

    def build_task_context(self, state: SwarmState) -> str:
        dest_name = state.get("destination_name", "Unknown Destination")
        dest_lat = state.get("destination_lat")
        dest_lon = state.get("destination_lon")
        custom_th = state.get("custom_thresholds")
        return (
            f"Relief Mission Destination: {dest_name} ({dest_lat}, {dest_lon}).\n"
            f"Active Threshold Overrides: {custom_th if custom_th else 'Default Indian NDRF Thresholds'}.\n"
            f"Objective: Scan all 20 corridor hubs, poll live disaster feeds, and register hazard zones."
        )

    def process_tool_result(
        self,
        state: SwarmState,
        tool_name: str,
        tool_result: Any,
        tool_args: Dict[str, Any],
    ) -> None:
        if tool_name == "tool_scan_weather_corridors":
            state["weather_scan"] = tool_result
            hazards: List[HazardInfo] = state.get("hazards", [])
            hazard_zones_db = state.get("hazard_zones_db", [])

            for hub_data in tool_result.get("hazardous_hubs", []):
                assessment = hub_data.get("assessment", {})
                weather = hub_data.get("weather", {})
                hazard_info: HazardInfo = {
                    "hub_name": hub_data["hub_name"],
                    "lat": hub_data["lat"],
                    "lon": hub_data["lon"],
                    "hazard_type": assessment.get("primary_hazard_type", "extreme_weather"),
                    "severity": assessment.get("severity", "high"),
                    "weather_data": weather,
                    "assessment": assessment,
                    "radius_km": HAZARD_RADIUS_KM,
                }

                # Register in database if not registered
                try:
                    reg = tool_register_hazard_zone(
                        name=f"Hazard near {hub_data['hub_name']}",
                        hazard_type=assessment.get("primary_hazard_type", "extreme_weather"),
                        severity=assessment.get("severity", "high"),
                        center_lat=hub_data["lat"],
                        center_lon=hub_data["lon"],
                        radius_km=HAZARD_RADIUS_KM,
                        source="open_meteo",
                    )
                    hazard_info["hazard_zone_id"] = reg.get("hazard_zone_id")
                    hazard_zones_db.append(reg)
                except Exception as e:
                    logger.warning(f"Could not register hazard zone for {hub_data['hub_name']}: {e}")

                hazards.append(hazard_info)

            state["hazards"] = hazards
            state["hazard_zones_db"] = hazard_zones_db

        elif tool_name == "tool_scan_live_disaster_feeds":
            hazards = state.get("hazards", [])
            hazard_zones_db = state.get("hazard_zones_db", [])
            disaster_events = tool_result if isinstance(tool_result, list) else []

            for ev in disaster_events:
                ev_lat = ev.get("lat")
                ev_lon = ev.get("lon")
                if ev_lat is not None and ev_lon is not None:
                    ev_title = ev.get("title", "Natural Hazard")
                    ev_type = ev.get("hazard_type", "disaster")
                    ev_radius = ev.get("radius_km", HAZARD_RADIUS_KM)
                    ev_source = ev.get("source", "USGS/NASA EONET")

                    try:
                        zone_res = tool_register_hazard_zone(
                            name=f"{ev_title} ({ev_source})",
                            hazard_type=ev_type,
                            severity=ev.get("severity", "high"),
                            center_lat=ev_lat,
                            center_lon=ev_lon,
                            radius_km=ev_radius,
                            source=ev_source,
                        )
                        hazard_zones_db.append(zone_res)
                        hazards.append({
                            "hub_name": ev_title,
                            "lat": ev_lat,
                            "lon": ev_lon,
                            "hazard_type": ev_type,
                            "severity": ev.get("severity", "high"),
                            "radius_km": ev_radius,
                            "source": ev_source,
                            "hazard_zone_id": zone_res.get("hazard_zone_id"),
                        })
                    except Exception as e:
                        logger.warning(f"Error registering live disaster event {ev_title}: {e}")

            state["hazards"] = hazards
            state["hazard_zones_db"] = hazard_zones_db

    def post_run_cleanup(self, state: SwarmState) -> None:
        if "hazards" not in state or state["hazards"] is None:
            state["hazards"] = []
        if "hazard_zones_db" not in state or state["hazard_zones_db"] is None:
            state["hazard_zones_db"] = []


# Singleton and convenience executor
_GLOBAL_TELEMETRY_AGENT = TelemetryAgent()

def run_telemetry_agent(state: SwarmState) -> SwarmState:
    """Entry point for Telemetry Agent execution."""
    return _GLOBAL_TELEMETRY_AGENT.run(state)

__all__ = ["TelemetryAgent", "run_telemetry_agent"]
