"""
Unit tests for Change 11 (Dual SQLite/PostgreSQL pooling + Alembic migrations),
Change 17 (GeoPy geodesic distances), and Change 19 (MILP multi-warehouse split optimization).
"""

import os
import sqlite3
from unittest.mock import MagicMock, patch

from geopy.distance import geodesic

from config import normalize_database_url
from database.db_manager import (
    _greedy_split_allocation,
    create_db_engine,
    get_db_diagnostics,
    get_nearest_warehouses,
    get_session,
    get_warehouse_inventory,
    optimize_multi_warehouse_fulfillment,
    reset_db,
    seed_all,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_normalize_database_url_variants():
    assert normalize_database_url("postgres://u:p@host:5432/db") == "postgresql://u:p@host:5432/db"
    assert normalize_database_url("postgresql://u:p@host:5432/db") == "postgresql://u:p@host:5432/db"
    assert normalize_database_url("postgresql+psycopg2://u:p@host/db") == "postgresql+psycopg2://u:p@host/db"
    assert normalize_database_url("sqlite:///supply_chain_swarm.db") == "sqlite:///supply_chain_swarm.db"


def test_sqlite_engine_and_diagnostics(tmp_path):
    temp_db = tmp_path / "diag_test.db"
    eng = create_db_engine(f"sqlite:///{temp_db}")
    diag = get_db_diagnostics(eng)
    assert diag["dialect"] == "sqlite"
    assert "Pool" in diag["pool_class"]
    eng.dispose()


def test_postgres_engine_pooling_configuration():
    with patch("database.db_manager.create_engine") as mock_create_engine:
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        eng = create_db_engine("postgres://user:pass@pg-host:5432/relief")
        assert eng is mock_engine
        mock_create_engine.assert_called_once()
        _, kwargs = mock_create_engine.call_args
        assert kwargs["pool_pre_ping"] is True
        assert kwargs["pool_size"] >= 1
        assert kwargs["max_overflow"] >= 1
        assert kwargs["pool_recycle"] > 0


def test_alembic_upgrade_head_creates_all_schema_tables(tmp_path, monkeypatch):
    from alembic.config import Config

    from alembic import command

    alembic_db = tmp_path / "alembic_schema.db"
    monkeypatch.setenv("ALEMBIC_DATABASE_URL", f"sqlite:///{alembic_db}")

    alembic_ini = os.path.join(PROJECT_ROOT, "alembic.ini")
    cfg = Config(alembic_ini)
    cfg.set_main_option("script_location", os.path.join(PROJECT_ROOT, "alembic"))

    command.upgrade(cfg, "head")

    conn = sqlite3.connect(str(alembic_db))
    try:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        table_names = {r[0] for r in rows}
    finally:
        conn.close()

    expected_tables = {
        "warehouses",
        "inventory",
        "fleet",
        "dispatches",
        "telemetry_logs",
        "hazard_zones",
        "alembic_version",
    }
    assert expected_tables.issubset(table_names)


def test_geopy_geodesic_nearest_warehouses(db_session):
    chennai_lat, chennai_lon = 13.0827, 80.2707
    nearest = get_nearest_warehouses(db_session, chennai_lat, chennai_lon, limit=5)
    assert len(nearest) == 5

    # Verify tuple ordering matches exact WGS-84 geodesic distance
    distances = [dist_km for _, dist_km in nearest]
    assert distances == sorted(distances)

    top_wh, top_dist_km = nearest[0]
    expected_km = geodesic((chennai_lat, chennai_lon), (top_wh.latitude, top_wh.longitude)).km
    assert abs(top_dist_km - expected_km) <= 0.1


def test_milp_and_greedy_multi_warehouse_split_optimization():
    reset_db()
    seed_all()
    session = get_session()
    try:
        # Surge demand larger than any single depot can fulfill alone
        surge_request = {
            "medical_kits": 450,
            "water_purification": 350,
            "food_rations": 900,
        }
        plan = optimize_multi_warehouse_fulfillment(
            session,
            dest_lat=17.6868,
            dest_lon=83.2185,
            requested_items=surge_request,
            max_depots=4,
        )

        assert plan["is_split_fulfillment"] is True
        assert plan["depot_count"] >= 2
        assert plan["fulfillment_rate_pct"] == 100.0
        assert plan["unfulfilled_items"] == {}

        # Verify exact item conservation across split depots
        for cat, qty in surge_request.items():
            allocated_for_cat = sum(
                alloc["allocated_items"].get(cat, 0) for alloc in plan["allocations"]
            )
            assert allocated_for_cat == qty

        # Validate distance-ordered greedy helper parity directly
        candidates = get_nearest_warehouses(session, 17.6868, 83.2185, limit=15)
        avail_matrix = {
            wh.id: {
                item.category: max(0, item.quantity - item.min_reserve)
                for item in get_warehouse_inventory(session, wh.id)
            }
            for wh, _ in candidates
        }
        greedy_allocs = _greedy_split_allocation(candidates, avail_matrix, surge_request)
        assert len(greedy_allocs) >= 2
        for cat, qty in surge_request.items():
            assert sum(depot.get(cat, 0) for depot in greedy_allocs.values()) == qty
    finally:
        session.close()
