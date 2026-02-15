"""Tests for Prometheus metrics endpoint (Phase 4, Session 3)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from bitbat.api.app import create_app
from bitbat.autonomous.db import AutonomousDB


@pytest.fixture()
def client() -> TestClient:
    return TestClient(create_app())


@pytest.fixture()
def db_with_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a temporary autonomous.db with sample data."""
    monkeypatch.chdir(tmp_path)
    db_path = tmp_path / "data" / "autonomous.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    from bitbat.autonomous.models import Base

    db = AutonomousDB(f"sqlite:///{db_path}")
    Base.metadata.create_all(db.engine)

    with db.session() as session:
        for i in range(10):
            pred = db.store_prediction(
                session,
                timestamp_utc=datetime(2024, 1, 1 + i, tzinfo=UTC).replace(tzinfo=None),
                predicted_direction="up",
                p_up=0.7,
                p_down=0.2,
                model_version="v1",
                freq="1h",
                horizon="4h",
            )
        # Realize 6 predictions (4 correct, 2 wrong)
        for pid in range(1, 7):
            db.realize_prediction(
                session,
                prediction_id=pid,
                actual_return=0.01 if pid <= 4 else -0.01,
                actual_direction="up" if pid <= 4 else "down",
            )
    return db_path


class TestMetricsEndpoint:
    def test_returns_200(self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_content_type_text(self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        resp = client.get("/metrics")
        assert "text/plain" in resp.headers["content-type"]

    def test_has_uptime_gauge(self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        text = client.get("/metrics").text
        assert "bitbat_uptime_seconds" in text

    def test_has_database_gauge(self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        text = client.get("/metrics").text
        assert "bitbat_database_available" in text

    def test_has_model_gauge(self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        text = client.get("/metrics").text
        assert "bitbat_model_available" in text

    def test_prometheus_format(self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        text = client.get("/metrics").text
        # Prometheus format: lines starting with # HELP, # TYPE, and metric lines
        assert "# HELP" in text
        assert "# TYPE" in text
        assert "gauge" in text


class TestMetricsWithData:
    def test_prediction_counts(self, client: TestClient, db_with_data: Path) -> None:
        text = client.get("/metrics").text
        assert "bitbat_predictions_total_30d" in text
        assert "bitbat_predictions_realized_30d" in text
        assert "bitbat_predictions_correct_30d" in text

    def test_hit_rate_present(self, client: TestClient, db_with_data: Path) -> None:
        text = client.get("/metrics").text
        assert "bitbat_hit_rate_30d" in text

    def test_database_shows_available(self, client: TestClient, db_with_data: Path) -> None:
        text = client.get("/metrics").text
        # The DB exists, so the gauge should be 1
        assert "bitbat_database_available 1" in text
