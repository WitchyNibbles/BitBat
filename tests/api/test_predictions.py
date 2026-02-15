"""Tests for prediction and analytics endpoints (Phase 4, Session 1)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from fastapi.testclient import TestClient

from bitbat.api.app import create_app
from bitbat.autonomous.db import AutonomousDB
from bitbat.autonomous.models import create_database_engine


@pytest.fixture()
def client() -> TestClient:
    app = create_app()
    return TestClient(app)


@pytest.fixture()
def db_with_predictions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a temporary autonomous.db with sample predictions."""
    db_path = tmp_path / "data" / "autonomous.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Make the predictions route find this DB
    monkeypatch.chdir(tmp_path)

    db = AutonomousDB(f"sqlite:///{db_path}")
    # Create tables
    from bitbat.autonomous.models import Base

    Base.metadata.create_all(db.engine)

    with db.session() as session:
        for i in range(5):
            db.store_prediction(
                session,
                timestamp_utc=datetime(2024, 1, 1 + i, tzinfo=UTC).replace(tzinfo=None),
                predicted_direction="up" if i % 2 == 0 else "down",
                p_up=0.7 if i % 2 == 0 else 0.3,
                p_down=0.2 if i % 2 == 0 else 0.6,
                model_version="v1",
                freq="1h",
                horizon="4h",
            )
        # Realize the first 3
        for pid in range(1, 4):
            db.realize_prediction(
                session,
                prediction_id=pid,
                actual_return=0.01 if pid % 2 == 1 else -0.005,
                actual_direction="up" if pid % 2 == 1 else "down",
            )

    return db_path


@pytest.fixture()
def model_on_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Train and save a tiny XGBoost model to the expected path."""
    monkeypatch.chdir(tmp_path)
    model_dir = tmp_path / "models" / "1h_4h"
    model_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    X = pd.DataFrame({"feat_a": rng.normal(size=30), "feat_b": rng.normal(size=30)})
    y = rng.choice([0, 1, 2], size=30)
    dtrain = xgb.DMatrix(X, label=y, feature_names=list(X.columns))
    booster = xgb.train({"objective": "multi:softprob", "num_class": 3, "max_depth": 2}, dtrain, 3)
    booster.save_model(str(model_dir / "xgb.json"))
    return model_dir / "xgb.json"


# ---------------------------------------------------------------------------
# Prediction endpoints
# ---------------------------------------------------------------------------


class TestLatestPrediction:
    def test_404_when_no_db(self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        resp = client.get("/predictions/latest")
        assert resp.status_code == 503

    def test_returns_prediction(self, client: TestClient, db_with_predictions: Path) -> None:
        resp = client.get("/predictions/latest?freq=1h&horizon=4h")
        assert resp.status_code == 200
        data = resp.json()
        assert "predicted_direction" in data
        assert "p_up" in data

    def test_404_wrong_config(self, client: TestClient, db_with_predictions: Path) -> None:
        resp = client.get("/predictions/latest?freq=1h&horizon=1h")
        assert resp.status_code == 404


class TestPredictionHistory:
    def test_returns_list(self, client: TestClient, db_with_predictions: Path) -> None:
        resp = client.get("/predictions/history?freq=1h&horizon=4h")
        assert resp.status_code == 200
        data = resp.json()
        assert "predictions" in data
        assert isinstance(data["predictions"], list)
        assert data["total"] == 5

    def test_limit_respected(self, client: TestClient, db_with_predictions: Path) -> None:
        resp = client.get("/predictions/history?freq=1h&horizon=4h&limit=2")
        data = resp.json()
        assert len(data["predictions"]) == 2
        assert data["total"] == 5

    def test_empty_history(self, client: TestClient, db_with_predictions: Path) -> None:
        resp = client.get("/predictions/history?freq=4h&horizon=24h")
        data = resp.json()
        assert data["total"] == 0


class TestPerformance:
    def test_returns_metrics(self, client: TestClient, db_with_predictions: Path) -> None:
        resp = client.get("/predictions/performance?freq=1h&horizon=4h")
        assert resp.status_code == 200
        data = resp.json()
        assert "hit_rate" in data
        assert data["realized_predictions"] == 3

    def test_empty_performance(self, client: TestClient, db_with_predictions: Path) -> None:
        resp = client.get("/predictions/performance?freq=4h&horizon=24h")
        data = resp.json()
        assert data["total_predictions"] == 0
        assert data["hit_rate"] is None


# ---------------------------------------------------------------------------
# Analytics endpoints
# ---------------------------------------------------------------------------


class TestFeatureImportance:
    def test_404_no_model(self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        resp = client.get("/analytics/feature-importance")
        assert resp.status_code == 404

    def test_returns_features(self, client: TestClient, model_on_disk: Path) -> None:
        resp = client.get("/analytics/feature-importance?freq=1h&horizon=4h")
        assert resp.status_code == 200
        data = resp.json()
        assert "features" in data
        assert isinstance(data["features"], list)

    def test_top_n_respected(self, client: TestClient, model_on_disk: Path) -> None:
        resp = client.get("/analytics/feature-importance?freq=1h&horizon=4h&top_n=1")
        data = resp.json()
        assert len(data["features"]) <= 1


class TestSystemStatus:
    def test_returns_status(self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        resp = client.get("/analytics/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "database_ok" in data
        assert "model_exists" in data

    def test_with_db(self, client: TestClient, db_with_predictions: Path) -> None:
        resp = client.get("/analytics/status?freq=1h&horizon=4h")
        data = resp.json()
        assert data["database_ok"] is True
        assert data["total_predictions"] == 5
