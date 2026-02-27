"""System endpoints — logs, retraining events, performance snapshots, ingestion status."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from bitbat.api.schemas import (
    PerformanceSnapshotEntry,
    PerformanceSnapshotsResponse,
    RetrainingEventEntry,
    RetrainingEventsResponse,
    SystemLogEntry,
    SystemLogsResponse,
)

router = APIRouter(prefix="/system", tags=["system"])

_DB_PATH = Path("data/autonomous.db")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_connection() -> sqlite3.Connection:
    """Return a sqlite3 connection, raising HTTP 503 if the DB is missing."""
    if not _DB_PATH.exists():
        raise HTTPException(status_code=503, detail="Database not available")
    return sqlite3.connect(str(_DB_PATH))


def _table_columns(con: sqlite3.Connection, table: str) -> set[str]:
    """Return available column names for a table."""
    rows = con.execute(f"PRAGMA table_info({table})").fetchall()  # noqa: S608
    return {str(row[1]) for row in rows if len(row) > 1}


def _first_available(columns: set[str], candidates: tuple[str, ...]) -> str | None:
    """Pick the first column name that exists."""
    for c in candidates:
        if c in columns:
            return c
    return None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/logs", response_model=SystemLogsResponse)
async def system_logs(
    limit: int = Query(50, ge=1, le=500),
    level: str | None = Query(None, description="Filter by log level (e.g. INFO, WARNING, ERROR)"),
) -> SystemLogsResponse:
    """Return recent entries from the system_logs table."""
    con = _get_connection()
    try:
        columns = _table_columns(con, "system_logs")
        ts_col = _first_available(columns, ("timestamp", "created_at"))
        if ts_col is None:
            return SystemLogsResponse(logs=[], total=0)

        service_expr = "service" if "service" in columns else "NULL"

        # Build count query
        count_sql = "SELECT COUNT(*) FROM system_logs"
        count_params: tuple[Any, ...] = ()
        if level is not None:
            count_sql += " WHERE level = ?"
            count_params = (level.upper(),)
        total: int = con.execute(count_sql, count_params).fetchone()[0]

        # Build data query
        select_sql = (
            f"SELECT {ts_col}, level, message, {service_expr} "  # noqa: S608
            f"FROM system_logs"
        )
        params: tuple[Any, ...] = ()
        if level is not None:
            select_sql += " WHERE level = ?"
            params = (level.upper(),)
        select_sql += f" ORDER BY {ts_col} DESC LIMIT ?"
        params = (*params, limit)

        rows = con.execute(select_sql, params).fetchall()
    finally:
        con.close()

    logs = [
        SystemLogEntry(
            timestamp=row[0],
            level=row[1] or "INFO",
            message=row[2] or "",
            service=row[3],
        )
        for row in rows
    ]
    return SystemLogsResponse(logs=logs, total=total)


@router.get("/retraining-events", response_model=RetrainingEventsResponse)
async def retraining_events(
    limit: int = Query(20, ge=1, le=100),
) -> RetrainingEventsResponse:
    """Return recent retraining events."""
    con = _get_connection()
    try:
        columns = _table_columns(con, "retraining_events")
        if not columns:
            return RetrainingEventsResponse(events=[], total=0)

        total: int = con.execute("SELECT COUNT(*) FROM retraining_events").fetchone()[0]

        # Build column expressions, handling optional columns
        def _col_or_null(name: str) -> str:
            return name if name in columns else f"NULL AS {name}"

        select_cols = [
            "id" if "id" in columns else "rowid AS id",
            _col_or_null("started_at"),
            _col_or_null("trigger_reason"),
            _col_or_null("status"),
            _col_or_null("old_model_version"),
            _col_or_null("new_model_version"),
            _col_or_null("cv_improvement"),
            _col_or_null("training_duration_seconds"),
        ]

        order_col = _first_available(columns, ("started_at", "id"))
        order_clause = f"ORDER BY {order_col} DESC" if order_col else ""

        sql = (
            f"SELECT {', '.join(select_cols)} "  # noqa: S608
            f"FROM retraining_events {order_clause} LIMIT ?"
        )
        rows = con.execute(sql, (limit,)).fetchall()
    finally:
        con.close()

    events = [
        RetrainingEventEntry(
            id=row[0],
            started_at=row[1],
            trigger_reason=row[2] or "unknown",
            status=row[3] or "unknown",
            old_model_version=row[4],
            new_model_version=row[5],
            cv_improvement=row[6],
            training_duration_seconds=row[7],
        )
        for row in rows
    ]
    return RetrainingEventsResponse(events=events, total=total)


@router.get("/performance-snapshots", response_model=PerformanceSnapshotsResponse)
async def performance_snapshots(
    limit: int = Query(20, ge=1, le=100),
) -> PerformanceSnapshotsResponse:
    """Return recent performance snapshots."""
    con = _get_connection()
    try:
        columns = _table_columns(con, "performance_snapshots")
        if not columns:
            return PerformanceSnapshotsResponse(snapshots=[])

        def _col_or_null(name: str) -> str:
            return name if name in columns else f"NULL AS {name}"

        select_cols = [
            _col_or_null("snapshot_time"),
            _col_or_null("model_version"),
            _col_or_null("hit_rate"),
            _col_or_null("total_predictions"),
            _col_or_null("sharpe_ratio"),
            _col_or_null("max_drawdown"),
        ]

        order_col = _first_available(columns, ("snapshot_time", "id"))
        order_clause = f"ORDER BY {order_col} DESC" if order_col else ""

        sql = (
            f"SELECT {', '.join(select_cols)} "  # noqa: S608
            f"FROM performance_snapshots {order_clause} LIMIT ?"
        )
        rows = con.execute(sql, (limit,)).fetchall()
    finally:
        con.close()

    snapshots = [
        PerformanceSnapshotEntry(
            snapshot_time=row[0],
            model_version=row[1] or "unknown",
            hit_rate=row[2],
            total_predictions=row[3] or 0,
            sharpe_ratio=row[4],
            max_drawdown=row[5],
        )
        for row in rows
    ]
    return PerformanceSnapshotsResponse(snapshots=snapshots)


@router.get("/ingestion-status")
async def ingestion_status() -> dict[str, Any]:
    """Return freshness information for ingested data files."""
    from bitbat.gui.widgets import get_ingestion_status

    return get_ingestion_status(Path("data"))
