"""
Integration tests for FastAPI Supplier and Relief Logistics REST Endpoints.
"""

from services.supplier_api import (
    StockCheckRequest,
    check_stock,
    list_dispatches,
    list_hazard_zones,
    list_warehouses,
    warehouse_inventory,
)


def test_list_warehouses_endpoint():
    data = list_warehouses()
    assert len(data) >= 20
    first = data[0]
    assert "name" in first
    assert "city" in first
    assert "latitude" in first
    assert "longitude" in first


def test_get_warehouse_inventory_endpoint():
    warehouses = list_warehouses()
    wh_id = warehouses[0]["id"]

    items = warehouse_inventory(wh_id)
    assert len(items) > 0
    assert "category" in items[0]
    assert "quantity" in items[0]


def test_check_stock_endpoint():
    warehouses = list_warehouses()
    wh_id = warehouses[0]["id"]

    req = StockCheckRequest(warehouse_id=wh_id, category="medical_kits", quantity=5)
    res_data = check_stock(req)
    assert "available" in res_data
    assert "available_above_reserve" in res_data
    assert res_data["requested"] == 5


def test_hazard_zones_endpoint():
    zones = list_hazard_zones()
    assert isinstance(zones, list)


def test_dispatches_endpoint():
    dispatches = list_dispatches()
    assert isinstance(dispatches, list)
