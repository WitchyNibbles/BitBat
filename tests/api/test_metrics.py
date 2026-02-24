"""Tests for Prometheus metrics endpoint (Phase 4, Session 3)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from tests.api.client import SyncASGIClient
from sqlalchemy import text

from bitbat.api.app import create_app
from bitbat.autonomous.db import AutonomousDB
from bitbat.autonomous.models import create_database_engine


@pytest.fixture()
def client() -> SyncASGIClient:
    return SyncASGIClient(create_app())


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
            db.store_prediction(
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


@pytest.fixture()
def incompatible_schema_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    db_path = tmp_path / "data" / "autonomous.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    engine = create_database_engine(f"sqlite:///{db_path}")
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
    return db_path


class TestMetricsEndpoint:
    def test_returns_200(
        self, client: SyncASGIClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_content_type_text(
        self, client: SyncASGIClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        resp = client.get("/metrics")
        assert "text/plain" in resp.headers["content-type"]

    def test_has_uptime_gauge(
        self, client: SyncASGIClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        text = client.get("/metrics").text
        assert "bitbat_uptime_seconds" in text

    def test_has_database_gauge(
        self, client: SyncASGIClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        text = client.get("/metrics").text
        assert "bitbat_database_available" in text

    def test_has_model_gauge(
        self, client: SyncASGIClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        text = client.get("/metrics").text
        assert "bitbat_model_available" in text

    def test_has_schema_gauges(
        self, client: SyncASGIClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        text = client.get("/metrics").text
        assert "bitbat_schema_compatible" in text
        assert "bitbat_schema_missing_columns" in text
        assert "bitbat_schema_auto_upgrade_possible" in text

    def test_prometheus_format(
        self, client: SyncASGIClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        text = client.get("/metrics").text
        # Prometheus format: lines starting with # HELP, # TYPE, and metric lines
        assert "# HELP" in text
        assert "# TYPE" in text
        assert "gauge" in text


class TestMetricsWithData:
    def test_prediction_counts(self, client: SyncASGIClient, db_with_data: Path) -> None:
        text = client.get("/metrics").text
        assert "bitbat_predictions_total_30d" in text
        assert "bitbat_predictions_realized_30d" in text
        assert "bitbat_predictions_correct_30d" in text

    def test_hit_rate_present(self, client: SyncASGIClient, db_with_data: Path) -> None:
        text = client.get("/metrics").text
        assert "bitbat_hit_rate_30d" in text

    def test_database_shows_available(self, client: SyncASGIClient, db_with_data: Path) -> None:
        text = client.get("/metrics").text
        # The DB exists, so the gauge should be 1
        assert "bitbat_database_available 1" in text

    def test_schema_shows_compatible(self, client: SyncASGIClient, db_with_data: Path) -> None:
        text = client.get("/metrics").text
        assert "bitbat_schema_compatible 1" in text
        assert "bitbat_schema_missing_columns 0" in text


class TestMetricsWithIncompatibleSchema:
    def test_schema_reports_incompatible(
        self,
        client: SyncASGIClient,
        incompatible_schema_db: Path,
    ) -> None:
        text = client.get("/metrics").text
        assert "bitbat_database_available 1" in text
        assert "bitbat_schema_compatible 0" in text
        assert "bitbat_schema_missing_columns 1" in text

    def test_prediction_gauges_are_not_emitted_when_schema_incompatible(
        self,
        client: SyncASGIClient,
        incompatible_schema_db: Path,
    ) -> None:
        text = client.get("/metrics").text
        assert "bitbat_predictions_total_30d" not in text
        assert "bitbat_predictions_realized_30d" not in text
        assert "bitbat_predictions_correct_30d" not in text
