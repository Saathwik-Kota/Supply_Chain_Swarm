"""Alembic migration environment supporting both SQLite and PostgreSQL."""

import os
import sys
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from config import DATABASE_URL, normalize_database_url
from database.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Prefer explicit ALEMBIC_DATABASE_URL, programmatic config override, or DATABASE_URL from config.py
ini_url = config.get_main_option("sqlalchemy.url")
if os.getenv("ALEMBIC_DATABASE_URL"):
    resolved_url = normalize_database_url(os.getenv("ALEMBIC_DATABASE_URL"))
elif ini_url and ini_url != "sqlite:///supply_chain_swarm.db":
    resolved_url = normalize_database_url(ini_url)
else:
    resolved_url = normalize_database_url(DATABASE_URL)

config.set_main_option("sqlalchemy.url", resolved_url)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=connection.dialect.name == "sqlite",
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
