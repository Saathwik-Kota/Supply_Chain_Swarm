"""Initial schema covering warehouses, inventory, fleet, dispatches, telemetry_logs, hazard_zones

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-11-11

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "0001_initial_schema"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "warehouses",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("city", sa.String(length=128), nullable=False),
        sa.Column("region", sa.String(length=64), nullable=False),
        sa.Column("latitude", sa.Float(), nullable=False),
        sa.Column("longitude", sa.Float(), nullable=False),
        sa.Column("capacity_tier", sa.String(length=16), nullable=False),
        sa.Column("address", sa.String(length=512), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )
    op.create_index("idx_warehouse_city", "warehouses", ["city"])
    op.create_index("idx_warehouse_region", "warehouses", ["region"])
    op.create_index("idx_warehouse_coords", "warehouses", ["latitude", "longitude"])

    op.create_table(
        "inventory",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("warehouse_id", sa.Integer(), nullable=False),
        sa.Column("category", sa.String(length=64), nullable=False),
        sa.Column("display_name", sa.String(length=128), nullable=False),
        sa.Column("quantity", sa.Integer(), nullable=False),
        sa.Column("unit", sa.String(length=32), nullable=False),
        sa.Column("weight_per_unit_kg", sa.Float(), nullable=False),
        sa.Column("priority", sa.Integer(), nullable=False),
        sa.Column("min_reserve", sa.Integer(), nullable=False),
        sa.Column("last_restocked", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["warehouse_id"], ["warehouses.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_inventory_warehouse", "inventory", ["warehouse_id"])
    op.create_index("idx_inventory_category", "inventory", ["category"])
    op.create_index(
        "idx_inventory_warehouse_category",
        "inventory",
        ["warehouse_id", "category"],
        unique=True,
    )

    op.create_table(
        "fleet",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("vehicle_id", sa.String(length=32), nullable=False),
        sa.Column("warehouse_id", sa.Integer(), nullable=False),
        sa.Column("vehicle_type", sa.String(length=32), nullable=False),
        sa.Column("display_name", sa.String(length=128), nullable=False),
        sa.Column("capacity_kg", sa.Float(), nullable=False),
        sa.Column("avg_speed_kmh", sa.Float(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("current_lat", sa.Float(), nullable=True),
        sa.Column("current_lon", sa.Float(), nullable=True),
        sa.Column("fuel_type", sa.String(length=20), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["warehouse_id"], ["warehouses.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("vehicle_id"),
    )
    op.create_index("idx_fleet_warehouse", "fleet", ["warehouse_id"])
    op.create_index("idx_fleet_status", "fleet", ["status"])

    op.create_table(
        "dispatches",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("dispatch_id", sa.String(length=64), nullable=False),
        sa.Column("source_warehouse_id", sa.Integer(), nullable=False),
        sa.Column("vehicle_id", sa.String(length=32), nullable=True),
        sa.Column("destination_name", sa.String(length=256), nullable=False),
        sa.Column("destination_lat", sa.Float(), nullable=False),
        sa.Column("destination_lon", sa.Float(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("items_manifest", sa.JSON(), nullable=True),
        sa.Column("total_weight_kg", sa.Float(), nullable=False),
        sa.Column("route_geometry", sa.Text(), nullable=True),
        sa.Column("route_distance_km", sa.Float(), nullable=True),
        sa.Column("route_duration_min", sa.Float(), nullable=True),
        sa.Column("eta_minutes", sa.Float(), nullable=True),
        sa.Column("hazard_avoided", sa.Boolean(), nullable=False),
        sa.Column("detour_reason", sa.Text(), nullable=True),
        sa.Column("agent_reasoning_log", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["source_warehouse_id"], ["warehouses.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("dispatch_id"),
    )
    op.create_index("idx_dispatch_status", "dispatches", ["status"])
    op.create_index("idx_dispatch_warehouse", "dispatches", ["source_warehouse_id"])
    op.create_index("idx_dispatch_created", "dispatches", ["created_at"])

    op.create_table(
        "telemetry_logs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("hub_name", sa.String(length=128), nullable=False),
        sa.Column("latitude", sa.Float(), nullable=False),
        sa.Column("longitude", sa.Float(), nullable=False),
        sa.Column("temperature_c", sa.Float(), nullable=True),
        sa.Column("humidity_pct", sa.Float(), nullable=True),
        sa.Column("precipitation_mm", sa.Float(), nullable=True),
        sa.Column("wind_speed_kmh", sa.Float(), nullable=True),
        sa.Column("wind_gusts_kmh", sa.Float(), nullable=True),
        sa.Column("weather_code", sa.Integer(), nullable=True),
        sa.Column("weather_description", sa.String(length=128), nullable=True),
        sa.Column("visibility_m", sa.Float(), nullable=True),
        sa.Column("is_hazardous", sa.Boolean(), nullable=False),
        sa.Column("hazard_type", sa.String(length=64), nullable=True),
        sa.Column("severity", sa.String(length=20), nullable=True),
        sa.Column("raw_api_response", sa.JSON(), nullable=True),
        sa.Column("recorded_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_telemetry_hub", "telemetry_logs", ["hub_name"])
    op.create_index("idx_telemetry_hazard", "telemetry_logs", ["is_hazardous"])
    op.create_index("idx_telemetry_time", "telemetry_logs", ["recorded_at"])

    op.create_table(
        "hazard_zones",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(length=256), nullable=False),
        sa.Column("hazard_type", sa.String(length=64), nullable=False),
        sa.Column("severity", sa.String(length=20), nullable=False),
        sa.Column("center_lat", sa.Float(), nullable=False),
        sa.Column("center_lon", sa.Float(), nullable=False),
        sa.Column("radius_km", sa.Float(), nullable=False),
        sa.Column("polygon_coords", sa.JSON(), nullable=True),
        sa.Column("source", sa.String(length=64), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("detected_at", sa.DateTime(), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_hazard_active", "hazard_zones", ["is_active"])
    op.create_index("idx_hazard_type", "hazard_zones", ["hazard_type"])
    op.create_index("idx_hazard_coords", "hazard_zones", ["center_lat", "center_lon"])


def downgrade() -> None:
    op.drop_table("hazard_zones")
    op.drop_table("telemetry_logs")
    op.drop_table("dispatches")
    op.drop_table("fleet")
    op.drop_table("inventory")
    op.drop_table("warehouses")

