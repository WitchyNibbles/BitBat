"""System endpoints — logs, retraining events, performance snapshots, training, settings."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException, Query

from bitbat.api.schemas import (
    PerformanceSnapshotEntry,
    PerformanceSnapshotsResponse,
    RetrainingEventEntry,
    RetrainingEventsResponse,
    SettingsResponse,
    SettingsUpdateRequest,
    SystemLogEntry,
    SystemLogsResponse,
    TrainingRequest,
    TrainingResponse,
)

router = APIRouter(prefix="/system", tags=["system"])

_DB_PATH = Path("data/autonomous.db")
_USER_CONFIG_PATH = Path("config/user_config.yaml")


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
# Task 1: System logs, retraining events, performance snapshots, ingestion
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


# ---------------------------------------------------------------------------
# Task 2: Training start and settings CRUD
# ---------------------------------------------------------------------------


@router.post("/training/start", response_model=TrainingResponse)
async def start_training(request: TrainingRequest) -> TrainingResponse:
    """Kick off a full training pipeline with the given preset."""
    from bitbat.gui.presets import list_presets

    available = list_presets()
    preset_key = request.preset.lower()
    if preset_key not in available:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown preset '{request.preset}'. "
            f"Available: {', '.join(sorted(available))}",
        )

    from bitbat.autonomous.orchestrator import one_click_train

    result = one_click_train(preset_name=preset_key)

    return TrainingResponse(
        status=result.get("status", "unknown"),
        model_version=result.get("model_version"),
        duration_seconds=result.get("duration_seconds"),
        error=result.get("error"),
    )


def _sorted_by_duration(freqs: set[str]) -> list[str]:
    """Sort frequency strings by their actual time duration."""
    import pandas as pd

    return sorted(freqs, key=lambda f: pd.to_timedelta(f))


def _valid_freqs() -> list[str]:
    """Return all supported frequencies sorted by duration."""
    from bitbat.timealign.bucket import _SUPPORTED_FREQUENCIES

    return _sorted_by_duration(_SUPPORTED_FREQUENCIES)


def _valid_horizons() -> list[str]:
    """Return valid horizon values (all supported frequencies except 1m)."""
    from bitbat.timealign.bucket import _SUPPORTED_FREQUENCIES

    return _sorted_by_duration(_SUPPORTED_FREQUENCIES - {"1m"})


def _load_defaults() -> dict[str, Any]:
    """Load default settings from default.yaml via the config loader."""
    from bitbat.config.loader import load_config

    return load_config()


@router.get("/settings", response_model=SettingsResponse)
async def get_settings() -> SettingsResponse:
    """Return current user settings, falling back to default.yaml values."""
    defaults = _load_defaults()
    valid_f = _valid_freqs()
    valid_h = _valid_horizons()

    if _USER_CONFIG_PATH.exists():
        try:
            raw = yaml.safe_load(_USER_CONFIG_PATH.read_text()) or {}
            # Merge: user config fields override defaults
            return SettingsResponse(
                preset=raw.get("preset", "custom"),
                freq=raw.get("freq", defaults.get("freq", "5m")),
                horizon=raw.get("horizon", defaults.get("horizon", "30m")),
                tau=raw.get("tau", defaults.get("tau", 0.01)),
                enter_threshold=raw.get(
                    "enter_threshold", defaults.get("enter_threshold", 0.6)
                ),
                valid_freqs=valid_f,
                valid_horizons=valid_h,
            )
        except Exception:  # noqa: BLE001, S110
            pass

    # No user config — fall back to default.yaml
    return SettingsResponse(
        preset="custom",
        freq=defaults.get("freq", "5m"),
        horizon=defaults.get("horizon", "30m"),
        tau=defaults.get("tau", 0.01),
        enter_threshold=defaults.get("enter_threshold", 0.6),
        valid_freqs=valid_f,
        valid_horizons=valid_h,
    )


@router.put("/settings", response_model=SettingsResponse)
async def update_settings(request: SettingsUpdateRequest) -> SettingsResponse:
    """Update user settings with freq/horizon validation against bucket.py."""
    from bitbat.timealign.bucket import _SUPPORTED_FREQUENCIES

    valid_f = _valid_freqs()
    valid_h = _valid_horizons()

    # Validate freq if provided
    if request.freq is not None and request.freq not in _SUPPORTED_FREQUENCIES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported frequency '{request.freq}'. Valid: {valid_f}",
        )

    # Validate horizon if provided (all supported except 1m)
    valid_horizon_set = _SUPPORTED_FREQUENCIES - {"1m"}
    if request.horizon is not None and request.horizon not in valid_horizon_set:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported horizon '{request.horizon}'. Valid: {valid_h}",
        )

    # Start from current settings as base
    current = await get_settings()
    base: dict[str, Any] = {
        "preset": current.preset,
        "freq": current.freq,
        "horizon": current.horizon,
        "tau": current.tau,
        "enter_threshold": current.enter_threshold,
    }

    # If a preset is specified, resolve its values as the new base
    if request.preset is not None:
        from bitbat.gui.presets import get_preset, list_presets

        preset_key = request.preset.lower()
        available = list_presets()
        if preset_key not in available:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown preset '{request.preset}'. "
                f"Available: {', '.join(sorted(available))}",
            )
        preset = get_preset(preset_key)
        base = {
            "preset": preset_key,
            "freq": preset.freq,
            "horizon": preset.horizon,
            "tau": preset.tau,
            "enter_threshold": preset.enter_threshold,
        }

    # Override with any explicitly provided values
    if request.freq is not None:
        base["freq"] = request.freq
    if request.horizon is not None:
        base["horizon"] = request.horizon
    if request.tau is not None:
        base["tau"] = request.tau
    if request.enter_threshold is not None:
        base["enter_threshold"] = request.enter_threshold

    # Write to config file
    _USER_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _USER_CONFIG_PATH.write_text(yaml.dump(base, default_flow_style=False, sort_keys=True))

    return SettingsResponse(valid_freqs=valid_f, valid_horizons=valid_h, **base)
