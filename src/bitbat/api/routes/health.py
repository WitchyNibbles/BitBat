"""Health-check endpoints."""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import APIRouter

from bitbat.api.defaults import _default_freq, _default_horizon
from bitbat.api.schemas import (
    DetailedHealthResponse,
    HealthResponse,
    SchemaReadinessDetails,
    ServiceStatus,
)
from bitbat.autonomous.schema_compat import audit_schema_compatibility, format_missing_columns
from bitbat.config.loader import resolve_models_dir

router = APIRouter(tags=["health"])

# Compute once at import time from config
_FREQ = _default_freq()
_HORIZON = _default_horizon()

_START_TIME = time.monotonic()
_VERSION = "0.1.0"


def _check_database() -> ServiceStatus:
    """Probe the autonomous SQLite database."""
    db_path = Path("data/autonomous.db")
    if db_path.exists():
        return ServiceStatus(name="database", status="ok")
    return ServiceStatus(name="database", status="unavailable", detail="autonomous.db not found")


def _check_schema_readiness() -> tuple[ServiceStatus, SchemaReadinessDetails]:
    """Audit schema compatibility without mutating the database."""
    db_path = Path("data/autonomous.db")
    if not db_path.exists():
        detail = "autonomous.db not found"
        return (
            ServiceStatus(name="schema_compatibility", status="unavailable", detail=detail),
            SchemaReadinessDetails(
                compatibility_state="unavailable",
                is_compatible=False,
                detail=detail,
            ),
        )

    database_url = f"sqlite:///{db_path}"
    try:
        report = audit_schema_compatibility(database_url=database_url)
    except Exception as exc:  # noqa: BLE001
        detail = f"schema audit failed: {exc}"
        return (
            ServiceStatus(name="schema_compatibility", status="error", detail=detail),
            SchemaReadinessDetails(
                compatibility_state="error",
                is_compatible=False,
                detail=detail,
            ),
        )

    if report.is_compatible:
        return (
            ServiceStatus(name="schema_compatibility", status="ok"),
            SchemaReadinessDetails(
                compatibility_state="compatible",
                is_compatible=True,
                can_auto_upgrade=report.can_auto_upgrade,
            ),
        )

    missing_columns = {
        table_name: list(columns) for table_name, columns in report.missing_columns.items()
    }
    missing_text = format_missing_columns(report)
    detail = f"missing required columns: {missing_text}"
    return (
        ServiceStatus(name="schema_compatibility", status="degraded", detail=detail),
        SchemaReadinessDetails(
            compatibility_state="incompatible",
            is_compatible=False,
            can_auto_upgrade=report.can_auto_upgrade,
            missing_columns=missing_columns,
            missing_columns_text=missing_text,
            detail=detail,
        ),
    )


def _check_model(freq: str = _FREQ, horizon: str = _HORIZON) -> ServiceStatus:
    """Check whether a trained XGBoost model exists."""
    model_path = resolve_models_dir() / f"{freq}_{horizon}" / "xgb.json"
    if model_path.exists():
        return ServiceStatus(name="model", status="ok")
    return ServiceStatus(
        name="model",
        status="unavailable",
        detail=f"No model at {model_path}",
    )


def _check_dataset(freq: str = _FREQ, horizon: str = _HORIZON) -> ServiceStatus:
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
async def health() -> HealthResponse:
    """Quick liveness probe — always returns ``ok``."""
    return HealthResponse(
        status="ok",
        version=_VERSION,
        uptime_seconds=round(time.monotonic() - _START_TIME, 1),
    )


@router.get("/health/detailed", response_model=DetailedHealthResponse)
async def health_detailed() -> DetailedHealthResponse:
    """Readiness probe — checks database, model, and dataset availability."""
    schema_service, schema_readiness = _check_schema_readiness()
    services = [_check_database(), schema_service, _check_model(), _check_dataset()]
    overall = "ok" if all(s.status == "ok" for s in services) else "degraded"
    return DetailedHealthResponse(
        status=overall,
        version=_VERSION,
        uptime_seconds=round(time.monotonic() - _START_TIME, 1),
        services=services,
        schema_readiness=schema_readiness,
    )
