"""
FastAPI router / service layer for live inventory, fleet, RAG, optimization, and WebSocket streams.

Exposes REST API and WebSocket endpoints backed by managed SQLAlchemy sessions (`Depends(get_db)`)
with interactive OpenAPI documentation available at `/docs` (Swagger UI) and `/redoc`.
"""

import asyncio
import logging
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import text
from sqlalchemy.orm import Session

from database.db_manager import (
    assign_vehicle,
    check_stock_availability,
    create_dispatch,
    create_hazard_zone,
    deduct_inventory,
    get_active_hazard_zones,
    get_all_dispatches,
    get_all_warehouses,
    get_available_vehicles,
    get_db,
    get_db_diagnostics,
    get_nearest_warehouses,
    get_regional_stock_summary,
    get_warehouse_by_id,
    get_warehouse_inventory,
    optimize_multi_warehouse_fulfillment,
    release_vehicle,
    session_scope,
)
from rag.retriever import get_rag_retriever
from services.cache_and_ratelimit import get_all_cache_and_ratelimit_stats
from services.circuit_breaker import get_all_breaker_statuses
from swarm.agent_framework import register_event_listener, unregister_event_listener

logger = logging.getLogger(__name__)

# ─── FastAPI App & OpenAPI Documentation Metadata ─────────────────────────────

OPENAPI_TAGS = [
    {"name": "Warehouses", "description": "Regional relief warehouse directory & geodesic proximity queries."},
    {"name": "Inventory", "description": "Reserve-safe stock lookups, ACID deductions, and MILP split optimization."},
    {"name": "Fleet", "description": "Emergency transport vehicle availability and assignment lifecycle."},
    {"name": "Dispatches", "description": "Committed multi-depot relief dispatch orders and manifests."},
    {"name": "Hazards", "description": "Active weather, seismic, and disaster hazard zone registration."},
    {"name": "RAG Knowledge Base", "description": "Hybrid TF-IDF/LSA retrieval over NDMA/NDRF SOP guidelines."},
    {"name": "Health & Resilience", "description": "DB pool diagnostics, circuit breakers, caches, and rate limiters."},
]

supplier_app = FastAPI(
    title="Emergency Supply Chain Swarm API",
    description=(
        "Production REST & WebSocket service powering autonomous disaster relief logistics, "
        "MILP multi-warehouse split fulfillment, NDMA RAG retrieval, and real-time agent streaming."
    ),
    version="2.0.0",
    license_info={"name": "MIT", "url": "https://opensource.org/licenses/MIT"},
    openapi_tags=OPENAPI_TAGS,
)


@contextmanager
def _use_db(db: Optional[Session]) -> Generator[Session, None, None]:
    """Use FastAPI-injected Session in HTTP requests, or open scoped session for direct calls."""
    if isinstance(db, Session):
        yield db
    else:
        with session_scope() as session:
            yield session


# ─── Pydantic Request/Response Models ────────────────────────────────────────


class StockCheckRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {"warehouse_id": 1, "category": "medical_kits", "quantity": 25}
        }
    )

    warehouse_id: int = Field(..., gt=0, description="Target regional warehouse primary ID")
    category: str = Field(..., description="Supply item category slug (e.g. medical_kits)")
    quantity: int = Field(..., gt=0, description="Requested unit count above safety reserve")


class DeductRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {"warehouse_id": 1, "category": "water_purification", "quantity": 10}
        }
    )

    warehouse_id: int = Field(..., gt=0, description="Source warehouse primary ID")
    category: str = Field(..., description="Supply item category slug")
    quantity: int = Field(..., gt=0, description="Quantity to deduct atomically")


class DispatchRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "source_warehouse_id": 2,
                "destination_name": "Chennai Coastal Emergency Zone",
                "destination_lat": 13.0827,
                "destination_lon": 80.2707,
                "items": {"medical_kits": 25, "food_rations": 50},
                "vehicle_id": "BLR-TRK-01",
                "hazard_avoided": True,
            }
        }
    )

    source_warehouse_id: int = Field(..., gt=0)
    destination_name: str
    destination_lat: float
    destination_lon: float
    items: Dict[str, int] = Field(..., description="Map of category slug -> quantity")
    vehicle_id: Optional[str] = None
    route_geometry: Optional[str] = None
    route_distance_km: Optional[float] = None
    route_duration_min: Optional[float] = None
    eta_minutes: Optional[float] = None
    hazard_avoided: bool = False
    detour_reason: Optional[str] = None
    agent_reasoning_log: Optional[str] = None


class HazardZoneRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "Cyclone Zone - Chennai",
                "hazard_type": "cyclone",
                "severity": "critical",
                "center_lat": 13.0827,
                "center_lon": 80.2707,
                "radius_km": 60.0,
                "source": "open_meteo",
            }
        }
    )

    name: str
    hazard_type: str
    severity: str = "high"
    center_lat: float
    center_lon: float
    radius_km: float = Field(default=50.0, gt=0)
    source: str = "open_meteo"


class SplitOptimizeRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "dest_lat": 17.6868,
                "dest_lon": 83.2185,
                "requested_items": {"medical_kits": 450, "water_purification": 350},
                "max_depots": 4,
            }
        }
    )

    dest_lat: float = Field(..., description="Disaster target latitude")
    dest_lon: float = Field(..., description="Disaster target longitude")
    requested_items: Dict[str, int] = Field(..., description="Category quantities to allocate")
    max_depots: int = Field(default=4, ge=1, le=10, description="Maximum depots to mobilize")


class RAGQueryRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "cyclone storm surge medical cold chain protocol",
                "disaster_type": "cyclone",
                "top_k": 3,
            }
        }
    )

    query: str = Field(..., min_length=2, description="Natural-language operational query")
    disaster_type: Optional[str] = Field(default=None, description="Optional hazard type boost")
    top_k: int = Field(default=3, ge=1, le=10)


# ─── Warehouse Endpoints ─────────────────────────────────────────────────────


@supplier_app.get(
    "/api/warehouses",
    tags=["Warehouses"],
    summary="List active logistics warehouses",
    description="Return all active regional logistics depots with geographic coordinates and tiers.",
)
def list_warehouses(db: Session = Depends(get_db)):
    with _use_db(db) as session:
        warehouses = get_all_warehouses(session)
        return [
            {
                "id": w.id,
                "name": w.name,
                "city": w.city,
                "region": w.region,
                "latitude": w.latitude,
                "longitude": w.longitude,
                "capacity_tier": w.capacity_tier,
                "is_active": w.is_active,
            }
            for w in warehouses
        ]


@supplier_app.get(
    "/api/warehouses/nearest",
    tags=["Warehouses"],
    summary="Find nearest warehouses by WGS-84 geodesic distance",
)
def nearest_warehouses(lat: float, lon: float, limit: int = 5, db: Session = Depends(get_db)):
    with _use_db(db) as session:
        results = get_nearest_warehouses(session, lat, lon, limit)
        return [
            {
                "warehouse": {
                    "id": wh.id,
                    "name": wh.name,
                    "city": wh.city,
                    "latitude": wh.latitude,
                    "longitude": wh.longitude,
                },
                "distance_km": round(dist, 2),
            }
            for wh, dist in results
        ]


@supplier_app.get(
    "/api/warehouses/{warehouse_id}",
    tags=["Warehouses"],
    summary="Get warehouse details by ID",
)
def get_warehouse(warehouse_id: int, db: Session = Depends(get_db)):
    with _use_db(db) as session:
        wh = get_warehouse_by_id(session, warehouse_id)
        if not wh:
            raise HTTPException(status_code=404, detail="Warehouse not found")
        return {
            "id": wh.id,
            "name": wh.name,
            "city": wh.city,
            "region": wh.region,
            "latitude": wh.latitude,
            "longitude": wh.longitude,
            "capacity_tier": wh.capacity_tier,
            "is_active": wh.is_active,
        }


# ─── Inventory Endpoints ─────────────────────────────────────────────────────


@supplier_app.get(
    "/api/inventory/{warehouse_id}",
    tags=["Inventory"],
    summary="Get warehouse stock ledger",
)
def warehouse_inventory(warehouse_id: int, db: Session = Depends(get_db)):
    with _use_db(db) as session:
        items = get_warehouse_inventory(session, warehouse_id)
        return [
            {
                "id": item.id,
                "category": item.category,
                "display_name": item.display_name,
                "quantity": item.quantity,
                "unit": item.unit,
                "weight_per_unit_kg": item.weight_per_unit_kg,
                "priority": item.priority,
                "min_reserve": item.min_reserve,
                "available_above_reserve": max(0, item.quantity - item.min_reserve),
                "last_restocked": item.last_restocked.isoformat() if item.last_restocked else None,
            }
            for item in items
        ]


@supplier_app.get(
    "/api/inventory/regional/{category}",
    tags=["Inventory"],
    summary="Query regional stock distribution for a category",
)
def regional_stock(category: str, db: Session = Depends(get_db)):
    with _use_db(db) as session:
        return get_regional_stock_summary(session, category)


@supplier_app.post(
    "/api/inventory/check",
    tags=["Inventory"],
    summary="Verify reserve-safe stock availability",
)
def check_stock(req: StockCheckRequest, db: Session = Depends(get_db)):
    with _use_db(db) as session:
        return check_stock_availability(session, req.warehouse_id, req.category, req.quantity)


@supplier_app.post(
    "/api/inventory/deduct",
    tags=["Inventory"],
    summary="Atomically deduct inventory with reserve guard",
)
def deduct_stock(req: DeductRequest, db: Session = Depends(get_db)):
    with _use_db(db) as session:
        result = deduct_inventory(session, req.warehouse_id, req.category, req.quantity)
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        return result


@supplier_app.post(
    "/api/inventory/optimize-split",
    tags=["Inventory"],
    summary="Solve MILP multi-warehouse split fulfillment",
    description="Allocates surge relief demand across multiple reserve-safe depots minimizing cargo-km.",
)
def api_optimize_multi_warehouse(req: SplitOptimizeRequest, db: Session = Depends(get_db)):
    with _use_db(db) as session:
        return optimize_multi_warehouse_fulfillment(
            session,
            req.dest_lat,
            req.dest_lon,
            req.requested_items,
            max_depots=req.max_depots,
        )


# ─── Fleet Endpoints ──────────────────────────────────────────────────────────


@supplier_app.get(
    "/api/fleet/{warehouse_id}",
    tags=["Fleet"],
    summary="List available transport vehicles at a depot",
)
def fleet_at_warehouse(
    warehouse_id: int, min_capacity_kg: float = 0, db: Session = Depends(get_db)
):
    with _use_db(db) as session:
        vehicles = get_available_vehicles(session, warehouse_id, min_capacity_kg)
        return [
            {
                "vehicle_id": v.vehicle_id,
                "vehicle_type": v.vehicle_type,
                "display_name": v.display_name,
                "capacity_kg": v.capacity_kg,
                "avg_speed_kmh": v.avg_speed_kmh,
                "status": v.status,
                "fuel_type": v.fuel_type,
            }
            for v in vehicles
        ]


@supplier_app.post(
    "/api/fleet/assign/{vehicle_id}",
    tags=["Fleet"],
    summary="Assign vehicle to an active dispatch mission",
)
def assign_fleet_vehicle(vehicle_id: str, db: Session = Depends(get_db)):
    with _use_db(db) as session:
        result = assign_vehicle(session, vehicle_id)
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        return result


@supplier_app.post(
    "/api/fleet/release/{vehicle_id}",
    tags=["Fleet"],
    summary="Release vehicle back to available pool",
)
def release_fleet_vehicle(vehicle_id: str, db: Session = Depends(get_db)):
    with _use_db(db) as session:
        result = release_vehicle(session, vehicle_id)
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        return result


# ─── Dispatch Endpoints ──────────────────────────────────────────────────────


@supplier_app.post(
    "/api/dispatches",
    tags=["Dispatches"],
    summary="Create persistent relief dispatch order",
)
def create_dispatch_order(req: DispatchRequest, db: Session = Depends(get_db)):
    with _use_db(db) as session:
        dispatch = create_dispatch(
            session,
            source_warehouse_id=req.source_warehouse_id,
            destination_name=req.destination_name,
            destination_lat=req.destination_lat,
            destination_lon=req.destination_lon,
            items_manifest=req.items,
            vehicle_id=req.vehicle_id,
            route_geometry=req.route_geometry,
            route_distance_km=req.route_distance_km,
            route_duration_min=req.route_duration_min,
            eta_minutes=req.eta_minutes,
            hazard_avoided=req.hazard_avoided,
            detour_reason=req.detour_reason,
            agent_reasoning_log=req.agent_reasoning_log,
        )
        return {
            "dispatch_id": dispatch.dispatch_id,
            "status": dispatch.status,
            "total_weight_kg": dispatch.total_weight_kg,
            "created_at": dispatch.created_at.isoformat(),
        }


@supplier_app.get(
    "/api/dispatches",
    tags=["Dispatches"],
    summary="List recent dispatch orders",
)
def list_dispatches(db: Session = Depends(get_db)):
    with _use_db(db) as session:
        dispatches = get_all_dispatches(session)
        return [
            {
                "dispatch_id": d.dispatch_id,
                "source_warehouse_id": d.source_warehouse_id,
                "vehicle_id": d.vehicle_id,
                "destination_name": d.destination_name,
                "destination_lat": d.destination_lat,
                "destination_lon": d.destination_lon,
                "status": d.status,
                "items_manifest": d.items_manifest,
                "total_weight_kg": d.total_weight_kg,
                "route_distance_km": d.route_distance_km,
                "route_duration_min": d.route_duration_min,
                "eta_minutes": d.eta_minutes,
                "hazard_avoided": d.hazard_avoided,
                "detour_reason": d.detour_reason,
                "created_at": d.created_at.isoformat(),
            }
            for d in dispatches
        ]


# ─── Hazard Zone & RAG Endpoints ──────────────────────────────────────────────


@supplier_app.get(
    "/api/hazard-zones",
    tags=["Hazards"],
    summary="List all active hazard zones",
)
def list_hazard_zones(db: Session = Depends(get_db)):
    with _use_db(db) as session:
        zones = get_active_hazard_zones(session)
        return [
            {
                "id": z.id,
                "name": z.name,
                "hazard_type": z.hazard_type,
                "severity": z.severity,
                "center_lat": z.center_lat,
                "center_lon": z.center_lon,
                "radius_km": z.radius_km,
                "source": z.source,
                "is_active": z.is_active,
                "detected_at": z.detected_at.isoformat(),
            }
            for z in zones
        ]


@supplier_app.post(
    "/api/hazard-zones",
    tags=["Hazards"],
    summary="Register active disaster hazard zone",
)
def add_hazard_zone(req: HazardZoneRequest, db: Session = Depends(get_db)):
    with _use_db(db) as session:
        zone = create_hazard_zone(
            session,
            name=req.name,
            hazard_type=req.hazard_type,
            severity=req.severity,
            center_lat=req.center_lat,
            center_lon=req.center_lon,
            radius_km=req.radius_km,
            source=req.source,
        )
        return {"id": zone.id, "name": zone.name, "is_active": zone.is_active}


@supplier_app.post(
    "/api/rag/query",
    tags=["RAG Knowledge Base"],
    summary="Query NDMA/NDRF relief SOP knowledge base",
    description="Performs hybrid TF-IDF + LSA vector retrieval over curated disaster logistics guidelines.",
)
def api_query_rag_guidelines(req: RAGQueryRequest):
    return get_rag_retriever().retrieve(
        query=req.query,
        disaster_type=req.disaster_type,
        top_k=req.top_k,
    )


# ─── Health & Resilience Telemetry ────────────────────────────────────────────


@supplier_app.get(
    "/api/health",
    tags=["Health & Resilience"],
    summary="Liveness & database connectivity probe",
)
def health_check(db: Session = Depends(get_db)):
    try:
        with _use_db(db) as session:
            session.execute(text("SELECT 1"))
            return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)},
        )


@supplier_app.get(
    "/api/health/resilience",
    tags=["Health & Resilience"],
    summary="Inspect connection pool, circuit breakers, caches, and rate limiters",
)
def resilience_health():
    return {
        "database": get_db_diagnostics(),
        "circuit_breakers": get_all_breaker_statuses(),
        "cache_and_rate_limits": get_all_cache_and_ratelimit_stats(),
    }


# ─── Real-Time WebSocket Streaming (/ws/swarm & /ws/events) ──────────────────


async def execute_swarm_over_websocket(websocket: WebSocket) -> None:
    """
    Accept a WebSocket launch request, execute the multi-agent swarm in a worker thread,
    and stream real-time agent lifecycle & evaluation events back to the client.
    """
    from swarm.orchestrator import run_swarm

    await websocket.accept()
    loop = asyncio.get_running_loop()
    event_queue: asyncio.Queue[Any] = asyncio.Queue()
    sentinel = object()

    def _enqueue_event(payload: Dict[str, Any]) -> None:
        try:
            loop.call_soon_threadsafe(event_queue.put_nowait, payload)
        except RuntimeError:
            pass

    register_event_listener(_enqueue_event)
    try:
        launch_cfg = await websocket.receive_json()
        await websocket.send_json({
            "event": "ack",
            "message": "Swarm execution accepted",
            "destination_name": launch_cfg.get("destination_name", "Emergency Zone"),
        })

        def _worker() -> Dict[str, Any]:
            try:
                return dict(
                    run_swarm(
                        destination_name=launch_cfg.get("destination_name", "Emergency Zone"),
                        destination_lat=float(launch_cfg.get("destination_lat", 13.0827)),
                        destination_lon=float(launch_cfg.get("destination_lon", 80.2707)),
                        requested_items=launch_cfg.get(
                            "requested_items", {"water_purification": 20, "food_rations": 50}
                        ),
                        custom_thresholds=launch_cfg.get("custom_thresholds"),
                    )
                )
            finally:
                try:
                    loop.call_soon_threadsafe(event_queue.put_nowait, sentinel)
                except RuntimeError:
                    pass

        worker_task = asyncio.create_task(asyncio.to_thread(_worker))

        while True:
            item = await event_queue.get()
            if item is sentinel:
                break
            await websocket.send_json(item)

        final_state = await worker_task
        await websocket.send_json({
            "event": "swarm_result",
            "swarm_status": final_state.get("swarm_status"),
            "total_latency_ms": final_state.get("total_latency_ms"),
            "evaluation_metrics": final_state.get("evaluation_metrics"),
            "multi_warehouse_plan": final_state.get("multi_warehouse_plan"),
            "rag_context": final_state.get("rag_context", []),
            "dispatch_manifest": final_state.get("dispatch_manifest"),
            "split_dispatches": final_state.get("split_dispatches", []),
        })
    except WebSocketDisconnect:
        logger.info("WebSocket /ws/swarm client disconnected cleanly.")
    finally:
        unregister_event_listener(_enqueue_event)


@supplier_app.websocket("/ws/swarm")
async def websocket_swarm_runner(websocket: WebSocket):
    """WebSocket endpoint to launch the swarm and stream real-time agent updates."""
    await execute_swarm_over_websocket(websocket)


@supplier_app.websocket("/ws/events")
async def websocket_event_hub(websocket: WebSocket):
    """Passive WebSocket broadcast hub streaming all ongoing swarm lifecycle events."""
    await websocket.accept()
    loop = asyncio.get_running_loop()
    event_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()

    def _enqueue_event(payload: Dict[str, Any]) -> None:
        try:
            loop.call_soon_threadsafe(event_queue.put_nowait, payload)
        except RuntimeError:
            pass

    register_event_listener(_enqueue_event)
    try:
        await websocket.send_json({"event": "connected", "channel": "/ws/events"})
        while True:
            recv_task = asyncio.create_task(websocket.receive_text())
            queue_task = asyncio.create_task(event_queue.get())
            done, pending = await asyncio.wait(
                {recv_task, queue_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            if recv_task in done:
                msg = recv_task.result()
                if msg.strip().lower() in ("close", "disconnect", "quit"):
                    break
            if queue_task in done:
                await websocket.send_json(queue_task.result())
    except WebSocketDisconnect:
        logger.info("WebSocket /ws/events subscriber disconnected cleanly.")
    finally:
        unregister_event_listener(_enqueue_event)
