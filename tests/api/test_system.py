"""Tests for DB-backed /system API routes."""

from __future__ import annotations

import asyncio
import time
import sqlite3
from pathlib import Path

import httpx
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


def test_training_start_uses_threadpool(
    client: SyncASGIClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "bitbat.common.presets.list_presets",
        lambda: {"balanced": object()},
    )

    calls: list[tuple[object, tuple[object, ...], dict[str, object]]] = []

    async def fake_run_in_threadpool(
        func: object,
        *args: object,
        **kwargs: object,
    ) -> dict[str, object]:
        calls.append((func, args, kwargs))
        return {
            "status": "success",
            "model_version": "v-threaded",
            "duration_seconds": 1.2,
        }

    def fake_train(*, preset_name: str) -> dict[str, object]:
        return {
            "status": "unexpected_direct_call",
            "model_version": preset_name,
        }

    monkeypatch.setattr("bitbat.api.routes.system.run_in_threadpool", fake_run_in_threadpool)
    monkeypatch.setattr("bitbat.autonomous.orchestrator.one_click_train", fake_train)

    response = client.request("POST", "/system/training/start", json={"preset": "balanced"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["model_version"] == "v-threaded"
    assert calls
    func, args, kwargs = calls[0]
    assert func is fake_train
    assert kwargs == {"preset_name": "balanced"}
    assert args == ()


def test_training_start_runs_off_event_loop_so_logs_can_refresh(
    db_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeDB:
        def list_system_logs(self, *, limit: int, level: str | None = None) -> dict[str, object]:
            del limit, level
            return {
                "logs": [
                    {
                        "timestamp": "2026-03-31T12:00:00",
                        "level": "INFO",
                        "message": "Training started",
                        "service": "training",
                    }
                ],
                "total": 1,
            }

    monkeypatch.setattr("bitbat.api.routes.system._DB_PATH", db_path)
    monkeypatch.setattr(
        "bitbat.api.routes.system._USER_CONFIG_PATH",
        tmp_path / "user_config.yaml",
    )
    monkeypatch.setattr("bitbat.api.routes.system._get_db", lambda: _FakeDB())
    monkeypatch.setattr(
        "bitbat.autonomous.orchestrator.one_click_train",
        lambda preset_name: (
            time.sleep(0.2),
            {
                "status": "success",
                "model_version": f"{preset_name}-v1",
                "duration_seconds": 0.2,
            },
        )[1],
    )

    app = create_app()

    async def _exercise() -> tuple[httpx.Response, httpx.Response, float]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            follow_redirects=True,
        ) as async_client:
            training_task = asyncio.create_task(
                async_client.post("/system/training/start", json={"preset": "balanced"})
            )
            await asyncio.sleep(0.05)
            start = time.perf_counter()
            logs_response = await async_client.get("/system/logs?limit=1")
            elapsed = time.perf_counter() - start
            training_response = await training_task
            return logs_response, training_response, elapsed

    logs_response, training_response, elapsed = asyncio.run(_exercise())

    assert logs_response.status_code == 200
    assert logs_response.json()["logs"][0]["message"] == "Training started"
    assert training_response.status_code == 200
    assert elapsed < 0.15
