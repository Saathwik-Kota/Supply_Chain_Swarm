"""
OpenAI & Gemini-compatible Tool/Function schemas for the agent swarm.
"""

from typing import Any, Dict, List

TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "tool_scan_weather_corridors": {
        "name": "tool_scan_weather_corridors",
        "description": "Scan all 20 Indian logistics corridor hubs using Open-Meteo API. Returns live weather and hazard assessment.",
        "parameters": {
            "type": "object",
            "properties": {
                "custom_thresholds": {
                    "type": "object",
                    "description": "Optional custom weather thresholds override (wind_speed_kmh, precipitation_mm_per_hour, etc.)",
                }
            },
            "required": [],
        },
    },
    "tool_scan_custom_location": {
        "name": "tool_scan_custom_location",
        "description": "Scan a specific latitude/longitude for weather telemetry and hazard condition.",
        "parameters": {
            "type": "object",
            "properties": {
                "lat": {"type": "number", "description": "Latitude coordinate"},
                "lon": {"type": "number", "description": "Longitude coordinate"},
                "name": {"type": "string", "description": "Location description or name"},
            },
            "required": ["lat", "lon"],
        },
    },
    "tool_scan_live_disaster_feeds": {
        "name": "tool_scan_live_disaster_feeds",
        "description": "Poll live seismic (USGS) and active disaster (NASA EONET) feeds across South Asia.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    "tool_register_hazard_zone": {
        "name": "tool_register_hazard_zone",
        "description": "Register an active hazard zone in the persistent database to prevent convoy entry.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name or identifier of hazard zone"},
                "hazard_type": {"type": "string", "description": "Type: cyclone, flood, earthquake, landslide, extreme_weather"},
                "severity": {"type": "string", "description": "Severity: low, moderate, high, critical"},
                "center_lat": {"type": "number", "description": "Latitude of epicenter"},
                "center_lon": {"type": "number", "description": "Longitude of epicenter"},
                "radius_km": {"type": "number", "description": "Radius of impact in km (default 50.0)"},
                "source": {"type": "string", "description": "Telemetry source (e.g. open_meteo, usgs, nasa_eonet)"},
            },
            "required": ["name", "hazard_type", "severity", "center_lat", "center_lon"],
        },
    },
    "tool_get_active_hazard_zones_from_db": {
        "name": "tool_get_active_hazard_zones_from_db",
        "description": "Retrieve all active hazard zones currently saved in the database.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    "tool_predict_relief_demand": {
        "name": "tool_predict_relief_demand",
        "description": "Use the Multi-Output RandomForest ML model to forecast emergency relief supply demand based on disaster parameters.",
        "parameters": {
            "type": "object",
            "properties": {
                "disaster_type": {"type": "string", "description": "Disaster type (cyclone, flood, earthquake, landslide, etc.)"},
                "severity": {"type": "string", "description": "Severity level (low, moderate, high, critical)"},
                "radius_km": {"type": "number", "description": "Impact radius in km"},
                "affected_population": {"type": "integer", "description": "Estimated affected population count"},
                "month": {"type": "integer", "description": "Current month (1-12)"},
                "rainfall_mm": {"type": "number", "description": "Rainfall intensity in mm/hr"},
                "wind_kmh": {"type": "number", "description": "Wind speed in km/h"},
            },
            "required": ["disaster_type", "severity"],
        },
    },
    "tool_find_nearest_warehouses": {
        "name": "tool_find_nearest_warehouses",
        "description": "Query database for closest regional warehouses to the given latitude and longitude coordinates.",
        "parameters": {
            "type": "object",
            "properties": {
                "lat": {"type": "number", "description": "Destination latitude"},
                "lon": {"type": "number", "description": "Destination longitude"},
                "limit": {"type": "integer", "description": "Number of warehouses to return (default 7)"},
            },
            "required": ["lat", "lon"],
        },
    },
    "tool_check_warehouse_inventory": {
        "name": "tool_check_warehouse_inventory",
        "description": "Check stock levels and minimum reserve compliance at a specific warehouse for requested items.",
        "parameters": {
            "type": "object",
            "properties": {
                "warehouse_id": {"type": "integer", "description": "Warehouse database ID"},
                "requested_items": {"type": "object", "description": "Dictionary mapping item category to requested count"},
            },
            "required": ["warehouse_id", "requested_items"],
        },
    },
    "tool_get_warehouse_full_inventory": {
        "name": "tool_get_warehouse_full_inventory",
        "description": "Fetch the full inventory breakdown and reserve levels for a specific warehouse.",
        "parameters": {
            "type": "object",
            "properties": {
                "warehouse_id": {"type": "integer", "description": "Warehouse database ID"},
            },
            "required": ["warehouse_id"],
        },
    },
    "tool_get_available_fleet": {
        "name": "tool_get_available_fleet",
        "description": "Query available transport vehicles (trucks, vans, helicopters) stationed at a warehouse with minimum payload capacity.",
        "parameters": {
            "type": "object",
            "properties": {
                "warehouse_id": {"type": "integer", "description": "Warehouse database ID"},
                "min_capacity_kg": {"type": "number", "description": "Minimum payload capacity required in kg"},
            },
            "required": ["warehouse_id"],
        },
    },
    "tool_get_route": {
        "name": "tool_get_route",
        "description": "Query live OSRM routing engine for the primary driving route between origin and destination.",
        "parameters": {
            "type": "object",
            "properties": {
                "origin_lat": {"type": "number", "description": "Origin latitude"},
                "origin_lon": {"type": "number", "description": "Origin longitude"},
                "dest_lat": {"type": "number", "description": "Destination latitude"},
                "dest_lon": {"type": "number", "description": "Destination longitude"},
            },
            "required": ["origin_lat", "origin_lon", "dest_lat", "dest_lon"],
        },
    },
    "tool_check_route_hazard_collision": {
        "name": "tool_check_route_hazard_collision",
        "description": "Check if a route polyline collides with any registered hazard zones using Shapely geometric spatial intersection.",
        "parameters": {
            "type": "object",
            "properties": {
                "route_data": {"type": "object", "description": "Route dictionary containing polyline and points"},
                "hazard_zones": {"type": "array", "items": {"type": "object"}, "description": "List of active hazard zones"},
            },
            "required": ["route_data", "hazard_zones"],
        },
    },
    "tool_assess_route_risk": {
        "name": "tool_assess_route_risk",
        "description": "Evaluate route transit risk and detect meteorological anomalies using the ML GradientBoosting + IsolationForest model.",
        "parameters": {
            "type": "object",
            "properties": {
                "wind_speed_kmh": {"type": "number", "description": "Wind speed along corridor"},
                "wind_gusts_kmh": {"type": "number", "description": "Wind gusts along corridor"},
                "precipitation_mm": {"type": "number", "description": "Rainfall rate mm/hr"},
                "temperature_c": {"type": "number", "description": "Temperature in celsius"},
                "humidity_pct": {"type": "number", "description": "Relative humidity percentage"},
                "visibility_m": {"type": "number", "description": "Visibility in meters"},
                "is_severe_wmo": {"type": "integer", "description": "1 if severe WMO code, 0 otherwise"},
                "distance_to_hazard_km": {"type": "number", "description": "Distance from closest route point to hazard center"},
                "route_distance_km": {"type": "number", "description": "Total route distance in km"},
            },
            "required": ["wind_speed_kmh", "wind_gusts_kmh", "precipitation_mm"],
        },
    },
    "tool_find_safe_route": {
        "name": "tool_find_safe_route",
        "description": "Autonomous self-correcting route solver. Queries OSRM, checks hazard collisions, calculates detour waypoints outside hazard buffers, and validates clearance.",
        "parameters": {
            "type": "object",
            "properties": {
                "origin_lat": {"type": "number", "description": "Origin latitude"},
                "origin_lon": {"type": "number", "description": "Origin longitude"},
                "dest_lat": {"type": "number", "description": "Destination latitude"},
                "dest_lon": {"type": "number", "description": "Destination longitude"},
                "hazard_zones": {"type": "array", "items": {"type": "object"}, "description": "List of active hazard zones to avoid"},
            },
            "required": ["origin_lat", "origin_lon", "dest_lat", "dest_lon", "hazard_zones"],
        },
    },
    "tool_deduct_inventory": {
        "name": "tool_deduct_inventory",
        "description": "Atomically commit inventory deductions in the database while strictly enforcing reserve floors.",
        "parameters": {
            "type": "object",
            "properties": {
                "warehouse_id": {"type": "integer", "description": "Warehouse database ID"},
                "items": {"type": "object", "description": "Dictionary of category to deducted quantity"},
            },
            "required": ["warehouse_id", "items"],
        },
    },
    "tool_assign_vehicle": {
        "name": "tool_assign_vehicle",
        "description": "Lock and assign a specific fleet vehicle in the database for the relief mission.",
        "parameters": {
            "type": "object",
            "properties": {
                "vehicle_id": {"type": "string", "description": "Vehicle registration/ID"},
            },
            "required": ["vehicle_id"],
        },
    },
    "tool_compute_carbon_and_costs": {
        "name": "tool_compute_carbon_and_costs",
        "description": "Compute environmental CO2 emissions, fuel burn, and monetary cost for the mission route (including detour delta).",
        "parameters": {
            "type": "object",
            "properties": {
                "distance_km": {"type": "number", "description": "Total route distance in km"},
                "vehicle_type": {"type": "string", "description": "Vehicle type: heavy_truck, medium_truck, light_van, helicopter"},
                "primary_distance_km": {"type": "number", "description": "Direct route distance without detour if applicable"},
            },
            "required": ["distance_km"],
        },
    },
    "tool_create_dispatch_order": {
        "name": "tool_create_dispatch_order",
        "description": "Create and commit the final emergency relief dispatch order with full manifest into the database.",
        "parameters": {
            "type": "object",
            "properties": {
                "source_warehouse_id": {"type": "integer", "description": "Origin warehouse ID"},
                "destination_name": {"type": "string", "description": "Destination name"},
                "destination_lat": {"type": "number", "description": "Destination latitude"},
                "destination_lon": {"type": "number", "description": "Destination longitude"},
                "items_manifest": {"type": "object", "description": "Items manifest dictionary"},
                "vehicle_id": {"type": "string", "description": "Assigned vehicle ID"},
                "route_geometry": {"type": "string", "description": "Encoded polyline geometry"},
                "route_distance_km": {"type": "number", "description": "Total route distance"},
                "route_duration_min": {"type": "number", "description": "Estimated duration in minutes"},
                "eta_minutes": {"type": "number", "description": "ETA in minutes"},
                "hazard_avoided": {"type": "boolean", "description": "Whether a detour was applied to avoid hazards"},
                "detour_reason": {"type": "string", "description": "Reason for detour if applied"},
                "agent_reasoning_log": {"type": "string", "description": "Consolidated agent reasoning audit trail"},
            },
            "required": [
                "source_warehouse_id",
                "destination_name",
                "destination_lat",
                "destination_lon",
                "items_manifest",
            ],
        },
    },
    "tool_optimize_multi_warehouse_fulfillment": {
        "name": "tool_optimize_multi_warehouse_fulfillment",
        "description": "Compute reserve-safe MILP optimal fulfillment split across multiple warehouses when one hub cannot fulfill 100%.",
        "parameters": {
            "type": "object",
            "properties": {
                "dest_lat": {"type": "number", "description": "Destination latitude"},
                "dest_lon": {"type": "number", "description": "Destination longitude"},
                "requested_items": {"type": "object", "description": "Requested supply categories and quantities"},
                "max_depots": {"type": "integer", "description": "Maximum depots in split plan"},
            },
            "required": ["dest_lat", "dest_lon", "requested_items"],
        },
    },
    "tool_retrieve_relief_guidelines": {
        "name": "tool_retrieve_relief_guidelines",
        "description": "Retrieve top matching NDMA/NDRF disaster relief SOPs and warehouse governance policies via semantic RAG.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language operational query"},
                "disaster_type": {"type": "string", "description": "Optional disaster type filter (e.g. cyclone, flood, earthquake)"},
                "top_k": {"type": "integer", "description": "Number of top matching documents to return"},
            },
            "required": ["query"],
        },
    },
}


def get_agent_tool_schemas(tool_names: List[str]) -> List[Dict[str, Any]]:
    """Return tool schemas in standard OpenAPI / Gemini compatible format."""
    return [TOOL_SCHEMAS[name] for name in tool_names if name in TOOL_SCHEMAS]

