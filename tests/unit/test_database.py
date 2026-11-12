"""
Unit tests for Database Manager and ORM CRUD operations.
"""

from database.db_manager import (
    assign_vehicle,
    check_stock_availability,
    create_hazard_zone,
    deduct_inventory,
    get_active_hazard_zones,
    get_all_warehouses,
    get_available_vehicles,
    get_nearest_warehouses,
    get_warehouse_inventory,
)


def test_warehouse_query(db_session):
    warehouses = get_all_warehouses(db_session)
    assert len(warehouses) >= 20, "Expected at least 20 seeded corridor hubs"


def test_nearest_warehouses(db_session):
    # Search near Chennai (13.08, 80.27)
    nearest = get_nearest_warehouses(db_session, lat=13.08, lon=80.27, limit=3)
    assert len(nearest) == 3
    # First warehouse should be closest (distance <= 50km)
    wh, dist = nearest[0]
    assert "Chennai" in wh.name or "Chennai" in wh.city or dist < 60.0
    # List must be ordered ascending by distance
    assert nearest[0][1] <= nearest[1][1] <= nearest[2][1]


def test_check_stock_and_reserve_enforcement(db_session):
    warehouses = get_all_warehouses(db_session)
    wh = warehouses[0]
    inv = get_warehouse_inventory(db_session, wh.id)
    assert len(inv) > 0

    item = inv[0]
    # Ensure sufficient stock above reserve for test
    if item.quantity <= item.min_reserve + 10:
        item.quantity = item.min_reserve + 20
        db_session.commit()

    # Check availability for reasonable quantity
    res_ok = check_stock_availability(db_session, wh.id, item.category, 5)
    assert res_ok["available"] is True

    # Check request exceeding quantity + reserve floor
    res_exceed = check_stock_availability(db_session, wh.id, item.category, 999999)
    assert res_exceed["available"] is False


def test_atomic_inventory_deduction(db_session):
    warehouses = get_all_warehouses(db_session)
    wh = warehouses[0]
    inv = get_warehouse_inventory(db_session, wh.id)
    item = inv[0]
    if item.quantity <= item.min_reserve + 10:
        item.quantity = item.min_reserve + 20
        db_session.commit()

    original_qty = item.quantity

    # Deduct 2 units
    deduct_res = deduct_inventory(db_session, wh.id, item.category, 2)
    assert deduct_res["success"] is True
    assert deduct_res["new_quantity"] == original_qty - 2

    # Verify reload from DB
    inv_after = get_warehouse_inventory(db_session, wh.id)
    item_after = next(i for i in inv_after if i.category == item.category)
    assert item_after.quantity == original_qty - 2


def test_vehicle_assignment(db_session):
    warehouses = get_all_warehouses(db_session)
    wh = warehouses[0]
    vehicles = get_available_vehicles(db_session, wh.id, min_capacity_kg=1000)
    assert len(vehicles) > 0

    target_v = vehicles[0]
    assign_res = assign_vehicle(db_session, target_v.vehicle_id)
    assert assign_res["success"] is True
    assert assign_res["vehicle_id"] == target_v.vehicle_id
    assert assign_res["capacity_kg"] == target_v.capacity_kg


def test_hazard_zone_persistence(db_session):
    zone = create_hazard_zone(
        db_session,
        name="Test Cyclone Michaung Zone",
        hazard_type="cyclone",
        severity="critical",
        center_lat=13.08,
        center_lon=80.27,
        radius_km=45.0,
        source="test_runner",
    )
    assert zone.id is not None

    active_zones = get_active_hazard_zones(db_session)
    zone_ids = [z.id for z in active_zones]
    assert zone.id in zone_ids
