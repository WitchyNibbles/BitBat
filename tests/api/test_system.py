"""Tests for DB-backed /system API routes."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from bitbat.api.app import create_app
from bitbat.autonomous.db import MonitorDatabaseError
from tests.api.client import SyncASGIClient

pytestmark = pytest.mark.integration


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "data" / "autonomous.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    sqlite3.connect(str(path)).close()
    return path


@pytest.fixture()
def client(
    db_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> SyncASGIClient:
    monkeypatch.setattr("bitbat.api.routes.system._DB_PATH", db_path)
    monkeypatch.setattr(
        "bitbat.api.routes.system._USER_CONFIG_PATH",
        tmp_path / "user_config.yaml",
    )
    return SyncASGIClient(create_app())


def test_system_logs_uses_autonomous_db_payload(
    client: SyncASGIClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "bitbat.autonomous.db.AutonomousDB.list_system_logs",
        lambda self, *, limit, level=None: {
            "logs": [
                {
                    "timestamp": "2026-03-12T12:00:00",
                    "level": "INFO",
                    "message": "hello",
                    "service": "monitoring_agent",
                }
            ],
            "total": 1,
        },
    )

    response = client.get("/system/logs")

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 1
    assert payload["logs"][0]["message"] == "hello"
    assert payload["logs"][0]["service"] == "monitoring_agent"


def test_retraining_events_uses_autonomous_db_payload(
    client: SyncASGIClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "bitbat.autonomous.db.AutonomousDB.list_retraining_events",
        lambda self, *, limit: {
            "events": [
                {
                    "id": 7,
                    "started_at": "2026-03-12T12:00:00",
                    "trigger_reason": "manual",
                    "status": "completed",
                    "old_model_version": "v1",
                    "new_model_version": "v2",
                    "cv_improvement": 0.03,
                    "training_duration_seconds": 12.5,
                }
            ],
            "total": 1,
        },
    )

    response = client.get("/system/retraining-events")

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 1
    assert payload["events"][0]["new_model_version"] == "v2"


def test_performance_snapshots_uses_autonomous_db_payload(
    client: SyncASGIClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "bitbat.autonomous.db.AutonomousDB.list_performance_snapshots",
        lambda self, *, limit: {
            "snapshots": [
                {
                    "snapshot_time": "2026-03-12T12:00:00",
                    "model_version": "v2",
                    "hit_rate": 0.61,
                    "total_predictions": 42,
                    "sharpe_ratio": 1.1,
                    "max_drawdown": -0.2,
                }
            ]
        },
    )

    response = client.get("/system/performance-snapshots")

    assert response.status_code == 200
    payload = response.json()
    assert payload["snapshots"][0]["model_version"] == "v2"
    assert payload["snapshots"][0]["total_predictions"] == 42


def test_system_logs_fail_fast_with_short_hint_line(
    client: SyncASGIClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "bitbat.autonomous.db.AutonomousDB.list_system_logs",
        lambda self, *, limit, level=None: (_ for _ in ()).throw(
            MonitorDatabaseError(
                step="system.logs",
                detail=(
                    "Database temporarily unavailable. "
                    "Circuit open after repeated lock retries."
                ),
                remediation="Retry shortly.",
                error_class="CircuitOpen",
                database_url="sqlite:///tmp/autonomous.db",
            )
        ),
    )

    response = client.get("/system/logs")

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "Hint: Retry shortly." in detail
    assert "CircuitOpen" not in detail
