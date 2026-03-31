"""Regression tests for the three dashboard live-update bugs.

Bug 1: Grimoire log not updating (system_logs table was never written to)
Bug 2: Crystal ball stuck at 10% (p_up/p_down not exposed in API)
Bug 3: Predictions not refreshing (only generated once at training time)
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from bitbat.autonomous.agent import MonitoringAgent
from bitbat.autonomous.db import AutonomousDB
from bitbat.autonomous.models import init_database

pytestmark = pytest.mark.integration


def _db_url(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path / 'dashboard_bugs.db'}"


def _seed_model(db: AutonomousDB) -> None:
    now = datetime.now(UTC).replace(tzinfo=None)
    with db.session() as session:
        db.store_model_version(
            session=session,
            version="v1.0.0",
            freq="1h",
            horizon="4h",
            training_start=now - timedelta(days=90),
            training_end=now,
            training_samples=1000,
            cv_score=0.62,
            features=["feat_x"],
            hyperparameters={"max_depth": 6},
            training_metadata={"tau": 0.01},
            is_active=True,
        )


def _seed_model_artifact(tmp_path: Path, freq: str = "1h", horizon: str = "4h") -> None:
    model_dir = tmp_path / "models" / f"{freq}_{horizon}"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "xgb.json").write_text("{}", encoding="utf-8")


def _disable_ingestion(agent: MonitoringAgent) -> None:
    agent._ingest_prices = lambda: None  # type: ignore[method-assign]
    agent._ingest_news = lambda: None  # type: ignore[method-assign]
    agent._ingest_auxiliary_data = lambda: None  # type: ignore[method-assign]


# --------------------------------------------------------------------------
# Bug 1: Grimoire log — system_logs table should receive entries from run_once
# --------------------------------------------------------------------------


class TestGrimoireLogWritten:
    """Verify that MonitoringAgent.run_once() writes to the system_logs table."""

    def test_run_once_writes_cycle_start_log(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        database_url = _db_url(tmp_path)
        init_database(database_url)
        db = AutonomousDB(database_url)
        _seed_model(db)
        _seed_model_artifact(tmp_path)
        monkeypatch.chdir(tmp_path)

        agent = MonitoringAgent(db, "1h", "4h")
        _disable_ingestion(agent)
        agent.validator.validate_all = lambda: {  # type: ignore[method-assign]
            "validated_count": 0,
            "correct_count": 0,
            "hit_rate": 0.0,
            "errors": [],
        }
        agent.predictor.predict_latest = lambda: {  # type: ignore[method-assign]
            "status": "no_prediction",
            "reason": "missing_model",
            "message": "Model artifact not found",
        }
        agent.drift_detector.check_drift = lambda: (  # type: ignore[method-assign]
            False,
            "No drift detected",
            {"hit_rate": 0.0},
        )
        agent.continuous_trainer.should_retrain = lambda: False  # type: ignore[method-assign]

        agent.run_once()

        # Verify system_logs table has entries
        logs_payload = db.list_system_logs(limit=10)
        assert logs_payload["total"] >= 1
        messages = [log["message"] for log in logs_payload["logs"]]
        # At minimum, the cycle-start log should be present
        assert any("cycle started" in m.lower() for m in messages)

    def test_run_once_logs_prediction_generated(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        database_url = _db_url(tmp_path)
        init_database(database_url)
        db = AutonomousDB(database_url)
        _seed_model(db)
        _seed_model_artifact(tmp_path)
        monkeypatch.chdir(tmp_path)

        agent = MonitoringAgent(db, "1h", "4h")
        _disable_ingestion(agent)
        agent.validator.validate_all = lambda: {  # type: ignore[method-assign]
            "validated_count": 0,
            "correct_count": 0,
            "hit_rate": 0.0,
            "errors": [],
        }
        agent.predictor.predict_latest = lambda: {  # type: ignore[method-assign]
            "status": "generated",
            "reason": "prediction_generated",
            "message": "Prediction generated and stored",
            "predicted_direction": "up",
        }
        agent.drift_detector.check_drift = lambda: (  # type: ignore[method-assign]
            False,
            "No drift detected",
            {"hit_rate": 0.0},
        )
        agent.continuous_trainer.should_retrain = lambda: False  # type: ignore[method-assign]

        agent.run_once()

        logs_payload = db.list_system_logs(limit=10)
        messages = [log["message"] for log in logs_payload["logs"]]
        assert any("prediction generated" in m.lower() for m in messages)

    def test_run_once_logs_drift_warning(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        database_url = _db_url(tmp_path)
        init_database(database_url)
        db = AutonomousDB(database_url)
        _seed_model(db)
        _seed_model_artifact(tmp_path)
        monkeypatch.chdir(tmp_path)

        agent = MonitoringAgent(db, "1h", "4h")
        _disable_ingestion(agent)
        agent.validator.validate_all = lambda: {  # type: ignore[method-assign]
            "validated_count": 0,
            "correct_count": 0,
            "hit_rate": 0.0,
            "errors": [],
        }
        agent.predictor.predict_latest = lambda: {  # type: ignore[method-assign]
            "status": "no_prediction",
            "reason": "missing_model",
            "message": "Model artifact not found",
        }
        agent.drift_detector.check_drift = lambda: (  # type: ignore[method-assign]
            True,
            "hit_rate_below_threshold",
            {"hit_rate": 0.25},
        )
        agent.continuous_trainer.should_retrain = lambda: False  # type: ignore[method-assign]

        agent.run_once()

        logs_payload = db.list_system_logs(limit=10)
        messages = [log["message"] for log in logs_payload["logs"]]
        levels = [log["level"] for log in logs_payload["logs"]]
        assert any("drift" in m.lower() for m in messages)
        assert "WARNING" in levels


# --------------------------------------------------------------------------
# Bug 2: Crystal ball — p_up/p_down available in /predictions/latest response
# --------------------------------------------------------------------------


class TestPredictionResponseProbabilities:
    """Verify p_up and p_down are included in the API prediction response."""

    def test_latest_prediction_includes_p_up_p_down(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from bitbat.api.app import create_app
        from tests.api.client import SyncASGIClient

        db_path = tmp_path / "data" / "autonomous.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        monkeypatch.chdir(tmp_path)

        db = AutonomousDB(f"sqlite:///{db_path}")
        with db.session() as session:
            db.store_prediction(
                session,
                timestamp_utc=datetime(2024, 6, 1, tzinfo=UTC).replace(tzinfo=None),
                predicted_direction="up",
                predicted_return=0.005,
                predicted_price=65000.0,
                p_up=0.72,
                p_down=0.18,
                model_version="v1",
                freq="1h",
                horizon="4h",
            )

        app = create_app()
        client = SyncASGIClient(app)

        resp = client.get("/predictions/latest?freq=1h&horizon=4h")
        assert resp.status_code == 200
        data = resp.json()

        # These fields must be present for the crystal ball fix
        assert "p_up" in data
        assert "p_down" in data
        assert data["p_up"] == pytest.approx(0.72, abs=0.01)
        assert data["p_down"] == pytest.approx(0.18, abs=0.01)


# --------------------------------------------------------------------------
# Bug 3: Orchestrator logs training events to system_logs
# --------------------------------------------------------------------------


class TestOrchestratorSystemLogs:
    """Verify one_click_train writes to system_logs when DB is available."""

    def test_training_log_entry_written(
        self,
        tmp_path: Path,
    ) -> None:
        """Directly test that db.log() creates a system_logs entry."""
        database_url = _db_url(tmp_path)
        init_database(database_url)
        db = AutonomousDB(database_url)

        with db.session() as session:
            db.log(session, "INFO", "training", "Training completed: 500 samples")

        logs_payload = db.list_system_logs(limit=5)
        assert logs_payload["total"] == 1
        assert "Training completed" in logs_payload["logs"][0]["message"]
        assert logs_payload["logs"][0]["level"] == "INFO"
        assert logs_payload["logs"][0]["service"] == "training"
