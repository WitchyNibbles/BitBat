"""Tests for health endpoints (Phase 4, Session 1)."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.api.client import SyncASGIClient
from sqlalchemy import text

from bitbat.api.app import create_app
from bitbat.autonomous.db import AutonomousDB
from bitbat.autonomous.models import create_database_engine


@pytest.fixture()
def client() -> SyncASGIClient:
    app = create_app()
    return SyncASGIClient(app)


def _create_legacy_prediction_outcomes(database_url: str) -> None:
    engine = create_database_engine(database_url)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS prediction_outcomes"))
        conn.execute(text(
            """
            CREATE TABLE prediction_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_utc DATETIME NOT NULL,
                prediction_timestamp DATETIME NOT NULL,
                predicted_direction VARCHAR(10) NOT NULL,
                p_up FLOAT,
                p_down FLOAT,
                p_flat FLOAT,
                predicted_return FLOAT,
                actual_return FLOAT,
                actual_direction VARCHAR(10),
                correct BOOLEAN,
                model_version VARCHAR(64) NOT NULL,
                freq VARCHAR(16) NOT NULL,
                horizon VARCHAR(16) NOT NULL,
                features_used JSON,
                created_at DATETIME NOT NULL,
                realized_at DATETIME
            )
            """
        ))
    engine.dispose()


def _schema_service(services: list[dict[str, str]]) -> dict[str, str]:
    return next(service for service in services if service["name"] == "schema_compatibility")


class TestHealthEndpoint:
    def test_returns_200(self, client: SyncASGIClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_status_ok(self, client: SyncASGIClient) -> None:
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_has_version(self, client: SyncASGIClient) -> None:
        data = client.get("/health").json()
        assert "version" in data
        assert isinstance(data["version"], str)

    def test_has_uptime(self, client: SyncASGIClient) -> None:
        data = client.get("/health").json()
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 0


class TestDetailedHealthEndpoint:
    def test_returns_200(self, client: SyncASGIClient) -> None:
        resp = client.get("/health/detailed")
        assert resp.status_code == 200

    def test_has_services_list(self, client: SyncASGIClient) -> None:
        data = client.get("/health/detailed").json()
        assert "services" in data
        assert isinstance(data["services"], list)

    def test_services_have_name_and_status(self, client: SyncASGIClient) -> None:
        data = client.get("/health/detailed").json()
        for svc in data["services"]:
            assert "name" in svc
            assert "status" in svc
            assert svc["status"] in ("ok", "degraded", "unavailable", "error")

    def test_schema_readiness_payload_exists(
        self,
        client: SyncASGIClient,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        data = client.get("/health/detailed").json()
        assert "schema_readiness" in data
        assert data["schema_readiness"]["compatibility_state"] == "unavailable"
        assert data["schema_readiness"]["is_compatible"] is False

    def test_four_services_checked(self, client: SyncASGIClient) -> None:
        data = client.get("/health/detailed").json()
        names = {s["name"] for s in data["services"]}
        assert names == {"database", "schema_compatibility", "model", "dataset"}

    def test_overall_status_is_string(self, client: SyncASGIClient) -> None:
        data = client.get("/health/detailed").json()
        assert data["status"] in ("ok", "degraded", "error")

    def test_schema_service_degraded_for_incompatible_schema(
        self,
        client: SyncASGIClient,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        db_path = tmp_path / "data" / "autonomous.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        _create_legacy_prediction_outcomes(f"sqlite:///{db_path}")

        data = client.get("/health/detailed").json()
        schema_service = _schema_service(data["services"])
        assert schema_service["status"] == "degraded"
        assert "predicted_price" in (schema_service.get("detail") or "")
        assert data["schema_readiness"]["compatibility_state"] == "incompatible"
        assert data["schema_readiness"]["is_compatible"] is False
        assert (
            "predicted_price"
            in data["schema_readiness"]["missing_columns"]["prediction_outcomes"]
        )
        assert data["schema_readiness"]["missing_columns_text"] is not None
        assert data["schema_readiness"]["missing_columns_text"] in schema_service["detail"]

    def test_schema_service_degraded_detail_matches_readiness_detail(
        self,
        client: SyncASGIClient,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        db_path = tmp_path / "data" / "autonomous.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        _create_legacy_prediction_outcomes(f"sqlite:///{db_path}")

        data = client.get("/health/detailed").json()
        schema_service = _schema_service(data["services"])
        assert schema_service["status"] == "degraded"
        assert schema_service["detail"] == data["schema_readiness"]["detail"]

    def test_schema_service_ok_for_compatible_schema(
        self,
        client: SyncASGIClient,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        db_path = tmp_path / "data" / "autonomous.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        AutonomousDB(f"sqlite:///{db_path}")

        data = client.get("/health/detailed").json()
        schema_service = _schema_service(data["services"])
        assert schema_service["status"] == "ok"
        assert data["schema_readiness"]["compatibility_state"] == "compatible"
        assert data["schema_readiness"]["is_compatible"] is True
