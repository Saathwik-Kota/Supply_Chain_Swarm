"""
Curated Disaster Relief & Humanitarian Logistics SOP Knowledge Base.

Contains structured documents modeled on:
  - National Disaster Management Authority (NDMA) Relief Guidelines
  - NDRF Field Operations & Corridor Contingency Protocols
  - CONCOR MMLP / ICD Warehouse Reserve & Fleet Dispatch SOPs
"""

from typing import Dict, List

RELIEF_SOP_DOCUMENTS: List[Dict[str, str]] = [
    {
        "doc_id": "SOP-NDMA-CYC-01",
        "title": "Coastal Cyclone & Storm Surge Relief Protocol (NDMA Chapter 4)",
        "category": "ndrf_protocol",
        "disaster_types": "cyclone,severe_storm,severe_storms,high_wind",
        "source": "NDMA National Cyclone Management Guidelines",
        "content": (
            "During Category 1-4 coastal cyclones along the Bay of Bengal (Chennai, Visakhapatnam, "
            "Bhubaneswar corridors), prioritize rapid mobilization of water purification units, medical first-aid "
            "kits, emergency shelter tents, and thermal blankets. If coastal feeder roads experience wind speeds "
            "exceeding 70 km/h or flooding, re-route heavy cargo trucks through inland bypass corridors at least "
            "50 km outside the storm eye perimeter, or dispatch cargo helicopters for high-priority medical drops."
        ),
    },
    {
        "doc_id": "SOP-NDMA-FLD-02",
        "title": "Monsoon Flood & Water Contamination Emergency SOP",
        "category": "ndrf_protocol",
        "disaster_types": "flood,heavy_precipitation",
        "source": "NDMA Urban & Riverine Flood Guidelines",
        "content": (
            "Severe precipitation (>30 mm/hr) rapidly compromises municipal potable water networks and creates "
            "waterborne pathogen risk within 12 hours. Allocation must emphasize water_purification units (minimum "
            "1 unit per 150 affected citizens), food_rations, and medical_kits. Stage convoys at elevated railside "
            "terminals (Fatuha, Amingaon, Majerhat) above flood contours and verify bridge clearance via live routing."
        ),
    },
    {
        "doc_id": "SOP-NDMA-EQK-03",
        "title": "Seismic Trauma, Shelter & Communications Restoration SOP",
        "category": "ndrf_protocol",
        "disaster_types": "earthquake",
        "source": "NDRF Urban Search & Rescue Manual",
        "content": (
            "Following earthquakes M4.5+, arterial highways and overpasses require inspection for structural collapse. "
            "Prioritize trauma medical_kits, tents_shelters, portable solar_generators for field triage lighting, "
            "and satellite/VHF communication_kits where cellular towers fail. Avoid routing heavy trucks within "
            "30-60 km of active seismic fault epicenters when viable parallel highways exist."
        ),
    },
    {
        "doc_id": "SOP-NDMA-LND-04",
        "title": "Ghats & Himalayan Landslide Cutoff Detour Protocol",
        "category": "ndrf_protocol",
        "disaster_types": "landslide",
        "source": "BRO & NDMA Mountain Corridor Directives",
        "content": (
            "Landslides in the Western Ghats (Pune-Mumbai expressway, Kochi-Coimbatore ghats) or Northeast hills "
            "sever primary mountain passes. Dispatch light delivery vans (<2000 kg) with shorter turning radii "
            "when detours traverse secondary state highways, or split heavy loads across flanking depots such "
            "as Chakan (Pune) and Whitefield (Bengaluru) to bypass blocked ghat sections."
        ),
    },
    {
        "doc_id": "SOP-WH-RES-05",
        "title": "Minimum Strategic Reserve Floor Governance Policy",
        "category": "warehouse_sop",
        "disaster_types": "all",
        "source": "CONCOR / NDRF National Buffer Stock Directive",
        "content": (
            "Every primary, secondary, and tertiary warehouse must preserve its category-specific min_reserve floor "
            "(e.g., 50 medical kits, 30 water purification units, 100 food ration packs) to protect local municipal "
            "surge readiness against secondary aftershocks or simultaneous regional crises. ACID deduction transactions "
            "must strictly reject withdrawals below min_reserve."
        ),
    },
    {
        "doc_id": "SOP-OPT-SPL-06",
        "title": "Multi-Warehouse Split Fulfillment Optimization SOP",
        "category": "logistics_sop",
        "disaster_types": "all",
        "source": "National Logistics Swarm Dispatch Manual",
        "content": (
            "When disaster surge demand exceeds the available-above-reserve stock of the single nearest depot, "
            "the Inventory Coordinator must trigger Multi-Depot Split Optimization using Mixed-Integer Linear "
            "Programming (MILP). The optimizer minimizes total cargo ton-kilometers plus depot mobilization overhead "
            "across regional hubs while achieving the highest feasible network-wide fulfillment rate."
        ),
    },
    {
        "doc_id": "SOP-FLT-CAR-07",
        "title": "Fleet Selection, Payload Matching & Carbon Stewardship SOP",
        "category": "fleet_sop",
        "disaster_types": "all",
        "source": "Green Emergency Logistics Protocol (Theme 2)",
        "content": (
            "Match total relief cargo weight against available depot fleet capacity with a strict safety margin. "
            "Select diesel light vans or medium/heavy trucks for clear surface corridors to minimize fuel burn "
            "(2.68 kg CO2/L diesel) compared to aviation fuel (2.52 kg CO2/L, ~150 L/100km). Reserve cargo "
            "helicopters strictly for coastal isolation, severed mountain passes, or critical life-saving detours."
        ),
    },
    {
        "doc_id": "SOP-RTE-AVD-08",
        "title": "Self-Correcting Hazard Perimeter Avoidance Standard",
        "category": "routing_sop",
        "disaster_types": "cyclone,flood,earthquake,landslide,high_wind,severe_weather_code",
        "source": "Autonomous Convoy Safety Directive",
        "content": (
            "All road trajectories returned by OSRM must undergo Shapely polygon intersection verification against "
            "active weather and seismic hazard zones (default 50 km safety radius). If any polyline segment penetrates "
            "a hazard boundary, the Dynamic Routing Agent must compute perpendicular safety waypoints outside the "
            "buffered perimeter and validate the detour trajectory before dispatch commitment."
        ),
    },
    {
        "doc_id": "SOP-COR-STH-09",
        "title": "Southern Peninsula Corridor Mutual-Aid Playbook (Chennai-Bengaluru-Hyderabad)",
        "category": "historical_report",
        "disaster_types": "cyclone,flood,heatwave",
        "source": "2023-2025 South India Cyclone After-Action Report",
        "content": (
            "Historical Cyclone Michaung response proved that staging stocks at CONCOR CFS Tondiarpet (Chennai) enables "
            "sub-15km intra-zone coastal relief when port approaches remain passable. When port inundation blocks eastern "
            "access, backup convoys from CONCOR MMLP Whitefield (Bengaluru, ~275km) and ICD Irugur (Coimbatore, ~416km) "
            "provide resilient western reinforcement along NH-48."
        ),
    },
    {
        "doc_id": "SOP-COR-NE-10",
        "title": "Northeast & Brahmaputra Monsoon Staging Playbook (Guwahati-Patna-Kolkata)",
        "category": "historical_report",
        "disaster_types": "flood,earthquake,landslide",
        "source": "NDRF 1st Battalion Brahmaputra Operations Review",
        "content": (
            "During severe Brahmaputra valley monsoon flooding, CONCOR MMLP Amingaon (Guwahati) serves as the primary "
            "staging anchor, backed by Majerhat Freight Terminal (Kolkata) and Fatuha Warehouse Complex (Patna). "
            "Helicopter airbridges and elevated railway sidings mitigate NH-27 submergence."
        ),
    },
    {
        "doc_id": "SOP-NDMA-HET-11",
        "title": "Extreme Heatwave & Cold-Chain Medical Preservation Protocol",
        "category": "ndrf_protocol",
        "disaster_types": "heatwave,extreme_heat",
        "source": "NDMA Heat Action Plan",
        "content": (
            "When ambient corridor temperatures exceed 44°C, medical_kits containing temperature-sensitive IV fluids, "
            "insulin, and vaccines require insulated transit bays and prioritized night-window dispatch. Boost water "
            "purification and oral rehydration allocations by 35% above baseline estimates."
        ),
    },
    {
        "doc_id": "SOP-AUD-ACD-12",
        "title": "ACID Transaction & Dispatch Manifest Traceability Standard",
        "category": "dispatch_sop",
        "disaster_types": "all",
        "source": "NDMA Supply Chain Audit Standard",
        "content": (
            "Before issuing a DSP dispatch order, the Autonomous Dispatch Coordinator must verify stock locks under "
            "database ACID isolation, confirm vehicle status transition from 'available' to 'dispatched', attach carbon "
            "and fuel expenditure accounting, and persist structured JSONL execution traces for post-operation auditing."
        ),
    },
]

