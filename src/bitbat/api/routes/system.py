"""System endpoints — logs, retraining events, performance snapshots, training, settings."""

from __future__ import annotations

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
from bitbat.autonomous.db import AutonomousDB, MonitorDatabaseError

router = APIRouter(prefix="/system", tags=["system"])

_DB_PATH = Path("data/autonomous.db")
_USER_CONFIG_PATH = Path("config/user_config.yaml")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_db() -> AutonomousDB:
    """Return an AutonomousDB handle, raising HTTP 503 if unavailable."""
    if not _DB_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "Database not available.\n"
                "Hint: run training or start the monitor to create data/autonomous.db."
            ),
        )
    try:
        return AutonomousDB(f"sqlite:///{_DB_PATH}")
    except MonitorDatabaseError as exc:
        raise _db_http_exception(exc) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Database temporarily unavailable.\n"
                "Hint: check database availability and monitor logs."
            ),
        ) from exc


def _db_http_exception(exc: MonitorDatabaseError) -> HTTPException:
    return HTTPException(
        status_code=503,
        detail=f"{exc.detail}\nHint: {exc.remediation}",
    )


# ---------------------------------------------------------------------------
# Task 1: System logs, retraining events, performance snapshots, ingestion
# ---------------------------------------------------------------------------


@router.get("/logs", response_model=SystemLogsResponse)
async def system_logs(
    limit: int = Query(50, ge=1, le=500),
    level: str | None = Query(None, description="Filter by log level (e.g. INFO, WARNING, ERROR)"),
) -> SystemLogsResponse:
    """Return recent entries from the system_logs table."""
    try:
        payload = _get_db().list_system_logs(limit=limit, level=level)
    except MonitorDatabaseError as exc:
        raise _db_http_exception(exc) from exc

    logs = [SystemLogEntry(**row) for row in payload["logs"]]
    return SystemLogsResponse(logs=logs, total=int(payload["total"]))


@router.get("/retraining-events", response_model=RetrainingEventsResponse)
async def retraining_events(
    limit: int = Query(20, ge=1, le=100),
) -> RetrainingEventsResponse:
    """Return recent retraining events."""
    try:
        payload = _get_db().list_retraining_events(limit=limit)
    except MonitorDatabaseError as exc:
        raise _db_http_exception(exc) from exc

    events = [RetrainingEventEntry(**row) for row in payload["events"]]
    return RetrainingEventsResponse(events=events, total=int(payload["total"]))


@router.get("/performance-snapshots", response_model=PerformanceSnapshotsResponse)
async def performance_snapshots(
    limit: int = Query(20, ge=1, le=100),
) -> PerformanceSnapshotsResponse:
    """Return recent performance snapshots."""
    try:
        payload = _get_db().list_performance_snapshots(limit=limit)
    except MonitorDatabaseError as exc:
        raise _db_http_exception(exc) from exc

    snapshots = [PerformanceSnapshotEntry(**row) for row in payload["snapshots"]]
    return PerformanceSnapshotsResponse(snapshots=snapshots)


@router.get("/ingestion-status")
async def ingestion_status() -> dict[str, Any]:
    """Return freshness information for ingested data files."""
    from bitbat.common.ingestion_status import get_ingestion_status

    return get_ingestion_status(Path("data"))


# ---------------------------------------------------------------------------
# Task 2: Training start and settings CRUD
# ---------------------------------------------------------------------------


@router.post("/training/start", response_model=TrainingResponse)
async def start_training(request: TrainingRequest) -> TrainingResponse:
    """Kick off a full training pipeline with the given preset."""
    from bitbat.common.presets import list_presets

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
                enter_threshold=raw.get("enter_threshold", defaults.get("enter_threshold", 0.6)),
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
        from bitbat.common.presets import get_preset, list_presets

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
