"""
Phase 4 Complete Integration Test.

Exercises the full production stack end-to-end:
  health → predictions → analytics → metrics → schema validation
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from fastapi.testclient import TestClient

from bitbat.api.app import create_app
from bitbat.api.schemas import (
    DetailedHealthResponse,
    FeatureImportanceResponse,
    HealthResponse,
    PerformanceResponse,
    PredictionListResponse,
    PredictionResponse,
    SystemStatusResponse,
)
from bitbat.autonomous.db import AutonomousDB
from bitbat.autonomous.models import Base


@pytest.fixture(scope="module")
def full_env(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Set up a complete environment: DB with predictions + trained model."""
    root = tmp_path_factory.mktemp("prod")

    # --- Database ---
    db_dir = root / "data"
    db_dir.mkdir()
    db_path = db_dir / "autonomous.db"
    db = AutonomousDB(f"sqlite:///{db_path}")
    Base.metadata.create_all(db.engine)

    with db.session() as session:
        # Store model version
        db.store_model_version(
            session,
            version="v1.0",
            freq="1h",
            horizon="4h",
            training_start=datetime(2024, 1, 1),
            training_end=datetime(2024, 6, 1),
            training_samples=1000,
            cv_score=0.62,
            features=["feat_a", "feat_b"],
            hyperparameters={"max_depth": 6},
            training_metadata=None,
            is_active=True,
        )
        # Store predictions
        rng = np.random.default_rng(42)
        for i in range(20):
            direction = "up" if rng.random() > 0.4 else "down"
            p_up = float(rng.uniform(0.3, 0.8))
            db.store_prediction(
                session,
                timestamp_utc=datetime(2024, 6, 1 + i, tzinfo=UTC).replace(tzinfo=None),
                predicted_direction=direction,
                p_up=p_up,
                p_down=max(0.0, 1.0 - p_up - 0.1),
                model_version="v1.0",
                freq="1h",
                horizon="4h",
            )
        # Realize 15
        for pid in range(1, 16):
            db.realize_prediction(
                session,
                prediction_id=pid,
                actual_return=float(rng.normal(0.001, 0.01)),
                actual_direction="up" if rng.random() > 0.4 else "down",
            )

    # --- Model ---
    model_dir = root / "models" / "1h_4h"
    model_dir.mkdir(parents=True)
    X = pd.DataFrame({"feat_a": rng.normal(size=50), "feat_b": rng.normal(size=50)})
    y = rng.choice([0, 1, 2], size=50)
    dtrain = xgb.DMatrix(X, label=y, feature_names=list(X.columns))
    booster = xgb.train({"objective": "multi:softprob", "num_class": 3, "max_depth": 2}, dtrain, 5)
    booster.save_model(str(model_dir / "xgb.json"))

    # --- Feature dataset ---
    feat_dir = root / "data" / "features" / "1h_4h"
    feat_dir.mkdir(parents=True)
    ds = pd.DataFrame({
        "timestamp_utc": pd.date_range("2024-01-01", periods=100, freq="1h"),
        "feat_a": rng.normal(size=100),
        "feat_b": rng.normal(size=100),
        "label": rng.choice(["up", "down", "flat"], size=100),
    })
    ds.to_parquet(feat_dir / "dataset.parquet", index=False)

    return root


@pytest.fixture()
def client(full_env: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.chdir(full_env)
    return TestClient(create_app())


class TestPhase4Integration:
    def test_health_ok(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        HealthResponse(**data)  # validates schema
        assert data["status"] == "ok"

    def test_detailed_health_all_ok(self, client: TestClient) -> None:
        resp = client.get("/health/detailed")
        data = resp.json()
        parsed = DetailedHealthResponse(**data)
        assert parsed.status == "ok"
        assert all(s.status == "ok" for s in parsed.services)

    def test_latest_prediction(self, client: TestClient) -> None:
        resp = client.get("/predictions/latest?freq=1h&horizon=4h")
        assert resp.status_code == 200
        data = resp.json()
        parsed = PredictionResponse(**data)
        assert parsed.model_version == "v1.0"

    def test_prediction_history(self, client: TestClient) -> None:
        resp = client.get("/predictions/history?freq=1h&horizon=4h")
        data = resp.json()
        parsed = PredictionListResponse(**data)
        assert parsed.total == 20
        assert len(parsed.predictions) == 20

    def test_prediction_performance(self, client: TestClient) -> None:
        resp = client.get("/predictions/performance?freq=1h&horizon=4h")
        data = resp.json()
        parsed = PerformanceResponse(**data)
        assert parsed.realized_predictions == 15
        assert parsed.hit_rate is not None
        assert 0.0 <= parsed.hit_rate <= 1.0

    def test_feature_importance(self, client: TestClient) -> None:
        resp = client.get("/analytics/feature-importance?freq=1h&horizon=4h")
        data = resp.json()
        parsed = FeatureImportanceResponse(**data)
        assert len(parsed.features) > 0

    def test_system_status(self, client: TestClient) -> None:
        resp = client.get("/analytics/status?freq=1h&horizon=4h")
        data = resp.json()
        parsed = SystemStatusResponse(**data)
        assert parsed.database_ok is True
        assert parsed.model_exists is True
        assert parsed.dataset_exists is True
        assert parsed.active_model_version == "v1.0"
        assert parsed.total_predictions == 20

    def test_prometheus_metrics(self, client: TestClient) -> None:
        resp = client.get("/metrics")
        assert resp.status_code == 200
        text = resp.text
        assert "bitbat_uptime_seconds" in text
        assert "bitbat_database_available 1" in text
        assert "bitbat_model_available 1" in text
        assert "bitbat_predictions_total_30d" in text
        assert "bitbat_hit_rate_30d" in text

    def test_openapi_schema_available(self, client: TestClient) -> None:
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert "paths" in schema
        assert "/health" in schema["paths"]
        assert "/predictions/latest" in schema["paths"]
        assert "/metrics" in schema["paths"]

    def test_full_api_roundtrip(self, client: TestClient) -> None:
        """End-to-end: health → predictions → performance → analytics → metrics."""
        # 1. Health
        assert client.get("/health").json()["status"] == "ok"

        # 2. Latest prediction
        latest = client.get("/predictions/latest?freq=1h&horizon=4h").json()
        assert latest["predicted_direction"] in ("up", "down", "flat")

        # 3. History
        history = client.get("/predictions/history?freq=1h&horizon=4h").json()
        assert history["total"] > 0

        # 4. Performance
        perf = client.get("/predictions/performance?freq=1h&horizon=4h").json()
        assert perf["realized_predictions"] > 0

        # 5. Feature importance
        fi = client.get("/analytics/feature-importance?freq=1h&horizon=4h").json()
        assert len(fi["features"]) > 0

        # 6. System status
        status = client.get("/analytics/status?freq=1h&horizon=4h").json()
        assert status["database_ok"] is True

        # 7. Metrics
        metrics = client.get("/metrics").text
        assert "bitbat_" in metrics
