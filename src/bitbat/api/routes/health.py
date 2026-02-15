"""Health-check endpoints."""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import APIRouter

from bitbat.api.schemas import DetailedHealthResponse, HealthResponse, ServiceStatus

router = APIRouter(tags=["health"])

_START_TIME = time.monotonic()
_VERSION = "0.1.0"


def _check_database() -> ServiceStatus:
    """Probe the autonomous SQLite database."""
    db_path = Path("data/autonomous.db")
    if db_path.exists():
        return ServiceStatus(name="database", status="ok")
    return ServiceStatus(name="database", status="unavailable", detail="autonomous.db not found")


def _check_model(freq: str = "1h", horizon: str = "4h") -> ServiceStatus:
    """Check whether a trained XGBoost model exists."""
    model_path = Path("models") / f"{freq}_{horizon}" / "xgb.json"
    if model_path.exists():
        return ServiceStatus(name="model", status="ok")
    return ServiceStatus(
        name="model",
        status="unavailable",
        detail=f"No model at {model_path}",
    )


def _check_dataset(freq: str = "1h", horizon: str = "4h") -> ServiceStatus:
    """Check whether the feature dataset exists."""
    ds_path = Path("data/features") / f"{freq}_{horizon}" / "dataset.parquet"
    if ds_path.exists():
        return ServiceStatus(name="dataset", status="ok")
    return ServiceStatus(
        name="dataset",
        status="unavailable",
        detail=f"No dataset at {ds_path}",
    )


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Quick liveness probe — always returns ``ok``."""
    return HealthResponse(
        status="ok",
        version=_VERSION,
        uptime_seconds=round(time.monotonic() - _START_TIME, 1),
    )


@router.get("/health/detailed", response_model=DetailedHealthResponse)
def health_detailed() -> DetailedHealthResponse:
    """Readiness probe — checks database, model, and dataset availability."""
    services = [_check_database(), _check_model(), _check_dataset()]
    overall = "ok" if all(s.status == "ok" for s in services) else "degraded"
    return DetailedHealthResponse(
        status=overall,
        version=_VERSION,
        uptime_seconds=round(time.monotonic() - _START_TIME, 1),
        services=services,
    )
