"""Shared database access utilities for the BitBat GUI."""

from __future__ import annotations

from pathlib import Path

from bitbat.autonomous.db import AutonomousDB


def get_db(db_path: Path) -> AutonomousDB | None:
    """Return a connected AutonomousDB instance or None if unavailable/missing."""
    if not db_path.exists():
        return None
    try:
        return AutonomousDB(
            f"sqlite:///{db_path}",
            allow_incompatible_schema=True,
        )
    except Exception:
        return None


def db_query(db_path: Path, sql: str, params: tuple = ()) -> list:
    """Run a SQL SELECT against the autonomous DB, returning rows or []."""
    db = get_db(db_path)
    if db is None:
        return []
    try:
        with db.engine.connect() as connection:
            return list(connection.exec_driver_sql(sql, params).fetchall())
    except Exception:
        return []
