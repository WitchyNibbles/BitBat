"""Shared heartbeat helpers for autonomous monitor status."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def heartbeat_path_for_db_url(db_url: str) -> Path:
    """Resolve the monitor heartbeat path from a database URL."""
    if db_url.startswith("sqlite:///"):
        db_path = Path(db_url.replace("sqlite:///", "", 1))
        return db_path.parent / "monitoring_agent_heartbeat.json"
    return Path("data") / "monitoring_agent_heartbeat.json"


def _normalize_failures(failures: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for failure in failures or []:
        if not isinstance(failure, dict):
            continue
        item = {
            "source": str(failure.get("source", "unknown")),
            "required": bool(failure.get("required", False)),
            "status": str(failure.get("status", "unknown")),
            "message": str(failure.get("message", "")),
        }
        details = failure.get("details")
        if isinstance(details, dict):
            item["details"] = details
        normalized.append(item)
    return normalized


def write_monitor_heartbeat(
    path: Path,
    *,
    status: str,
    freq: str,
    horizon: str,
    interval: int,
    db_url: str,
    config_source: str,
    config_path: str,
    error: str | None = None,
    cycle_prediction_state: str | None = None,
    cycle_prediction_reason: str | None = None,
    cycle_realization_state: str | None = None,
    cycle_diagnostic: str | None = None,
    cycle_ingestion_state: str | None = None,
    cycle_ingestion_failures: list[dict[str, Any]] | None = None,
) -> None:
    """Write a compact latest-cycle heartbeat payload."""
    payload: dict[str, Any] = {
        "status": status,
        "updated_at": datetime.now(UTC).replace(tzinfo=None).isoformat(),
        "freq": freq,
        "horizon": horizon,
        "config_source": config_source,
        "config_path": config_path,
        "interval_seconds": int(interval),
        "database_url": db_url,
    }
    if error is not None:
        payload["error"] = error
    if cycle_prediction_state is not None:
        payload["cycle_prediction_state"] = cycle_prediction_state
    if cycle_prediction_reason is not None:
        payload["cycle_prediction_reason"] = cycle_prediction_reason
    if cycle_realization_state is not None:
        payload["cycle_realization_state"] = cycle_realization_state
    if cycle_diagnostic is not None:
        payload["cycle_diagnostic"] = cycle_diagnostic
    if cycle_ingestion_state is not None:
        payload["cycle_ingestion_state"] = cycle_ingestion_state
    failures = _normalize_failures(cycle_ingestion_failures)
    if failures:
        payload["cycle_ingestion_failures"] = failures

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    except Exception:
        logger.debug("Failed to write heartbeat file", exc_info=True)
