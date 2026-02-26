from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import text

from bitbat.autonomous.agent import MonitoringAgent
from bitbat.autonomous.db import AutonomousDB, MonitorDatabaseError
from bitbat.autonomous.models import PerformanceSnapshot, init_database
from bitbat.autonomous.predictor import LivePredictor
from bitbat.autonomous.schema_compat import SchemaCompatibilityError


def _db_url(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path / 'agent.db'}"


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


def _seed_realized_predictions(db: AutonomousDB, *, total: int, correct_count: int) -> None:
    now = datetime.now(UTC).replace(tzinfo=None)
    with db.session() as session:
        for idx in range(total):
            pred = db.store_prediction(
                session=session,
                timestamp_utc=now - timedelta(hours=total + idx),
                predicted_direction="up",
                p_up=0.7,
                p_down=0.2,
                model_version="v1.0.0",
                freq="1h",
                horizon="4h",
            )
            actual_direction = "up" if idx < correct_count else "down"
            db.realize_prediction(
                session=session,
                prediction_id=pred.id,
                actual_return=0.01 if actual_direction == "up" else -0.01,
                actual_direction=actual_direction,
            )


def _create_legacy_prediction_outcomes(database_url: str) -> None:
    db = AutonomousDB(database_url, auto_upgrade_schema=False)
    with db.engine.begin() as conn:
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


def _disable_ingestion(agent: MonitoringAgent) -> None:
    agent._ingest_prices = lambda: None  # type: ignore[method-assign]
    agent._ingest_news = lambda: None  # type: ignore[method-assign]
    agent._ingest_auxiliary_data = lambda: None  # type: ignore[method-assign]


def _seed_model_artifact(tmp_path: Path, freq: str = "1h", horizon: str = "4h") -> None:
    model_dir = tmp_path / "models" / f"{freq}_{horizon}"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "xgb.json").write_text("{}", encoding="utf-8")


def _stub_runtime_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = {
        "data_dir": str(tmp_path / "data"),
        "enable_sentiment": False,
        "enable_garch": False,
        "enable_macro": False,
        "enable_onchain": False,
    }
    monkeypatch.setattr(
        "bitbat.autonomous.predictor.get_runtime_config",
        lambda: config,
    )
    monkeypatch.setattr(
        "bitbat.autonomous.predictor.load_config",
        lambda: config,
    )


def test_predict_latest_returns_missing_model_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_url = _db_url(tmp_path)
    init_database(database_url)
    db = AutonomousDB(database_url)
    _stub_runtime_config(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)

    predictor = LivePredictor(db=db, freq="1h", horizon="4h")
    result = predictor.predict_latest()

    assert result is not None
    assert result["status"] == "no_prediction"
    assert result["reason"] == "missing_model"


def test_predict_latest_returns_insufficient_data_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_url = _db_url(tmp_path)
    init_database(database_url)
    db = AutonomousDB(database_url)
    _stub_runtime_config(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)

    predictor = LivePredictor(db=db, freq="1h", horizon="4h")
    monkeypatch.setattr(predictor, "_load_model", lambda: object())

    few_bars = pd.DataFrame(
        {"close": [100.0] * 12},
        index=pd.date_range("2026-01-01", periods=12, freq="h"),
    )
    monkeypatch.setattr(
        "bitbat.autonomous.predictor._load_ingested_prices",
        lambda data_dir, freq: few_bars,
    )

    result = predictor.predict_latest()

    assert result is not None
    assert result["status"] == "no_prediction"
    assert result["reason"] == "insufficient_data"
    assert result["details"]["available_bars"] == 12
    assert result["details"]["required_bars"] == 30


def test_predict_latest_exposes_stable_diagnostic_reason_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_url = _db_url(tmp_path)
    init_database(database_url)
    db = AutonomousDB(database_url)
    _stub_runtime_config(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)

    predictor = LivePredictor(db=db, freq="1h", horizon="4h")
    result = predictor.predict_latest()

    assert result["status"] == "no_prediction"
    assert result["reason"] == "missing_model"
    assert result["diagnostic_reason"] == "missing_model"
    assert "model artifact" in result["diagnostic_message"].lower()


def test_run_once_reports_cycle_state_for_missing_model_no_predictions(
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

    result = agent.run_once()

    assert result["prediction_state"] == "none"
    assert result["prediction_reason"] == "missing_model"
    assert result["realization_state"] == "none"


def test_run_once_reports_cycle_state_for_pending_realizations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_url = _db_url(tmp_path)
    init_database(database_url)
    db = AutonomousDB(database_url)
    _seed_model(db)
    _seed_model_artifact(tmp_path)
    monkeypatch.chdir(tmp_path)

    with db.session() as session:
        db.store_prediction(
            session=session,
            timestamp_utc=datetime.now(UTC).replace(tzinfo=None),
            predicted_direction="up",
            p_up=0.6,
            p_down=0.3,
            model_version="v1.0.0",
            freq="1h",
            horizon="4h",
            predicted_return=0.01,
            predicted_price=100.0,
        )

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
        "reason": "duplicate_bar",
        "message": "Prediction already exists",
    }
    agent.drift_detector.check_drift = lambda: (  # type: ignore[method-assign]
        False,
        "No drift detected",
        {"hit_rate": 0.0},
    )
    agent.continuous_trainer.should_retrain = lambda: False  # type: ignore[method-assign]

    result = agent.run_once()

    assert result["prediction_state"] == "none"
    assert result["prediction_reason"] == "duplicate_bar"
    assert result["realization_state"] == "pending"


def test_monitoring_agent_integration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    database_url = _db_url(tmp_path)
    init_database(database_url)
    db = AutonomousDB(database_url)
    _seed_model(db)
    _seed_realized_predictions(db, total=40, correct_count=15)
    _seed_model_artifact(tmp_path)
    monkeypatch.chdir(tmp_path)

    agent = MonitoringAgent(db, "1h", "4h")
    agent.validator.validate_all = lambda: {  # type: ignore[method-assign]
        "validated_count": 0,
        "correct_count": 0,
        "hit_rate": 0.0,
        "errors": [],
    }
    agent.drift_detector.check_drift = lambda: (  # type: ignore[method-assign]
        True,
        "forced drift",
        {"hit_rate": 0.3},
    )
    agent.drift_detector.is_in_cooldown = lambda: False  # type: ignore[method-assign]
    agent.continuous_trainer.should_retrain = lambda: True  # type: ignore[method-assign]
    agent.continuous_trainer.retrain = lambda **_: {  # type: ignore[method-assign]
        "status": "completed",
        "new_model_version": "v2.0.0",
        "deployed": True,
    }
    _disable_ingestion(agent)

    result = agent.run_once()

    assert result["drift_detected"] is True
    assert result["retraining_triggered"] is True
    with db.session() as session:
        snapshot_count = session.query(PerformanceSnapshot).count()
    assert snapshot_count > 0


def test_no_drift_scenario(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    database_url = _db_url(tmp_path)
    init_database(database_url)
    db = AutonomousDB(database_url)
    _seed_model(db)
    _seed_realized_predictions(db, total=40, correct_count=35)
    _seed_model_artifact(tmp_path)
    monkeypatch.chdir(tmp_path)

    agent = MonitoringAgent(db, "1h", "4h")
    agent.validator.validate_all = lambda: {  # type: ignore[method-assign]
        "validated_count": 0,
        "correct_count": 0,
        "hit_rate": 0.0,
        "errors": [],
    }
    agent.drift_detector.check_drift = lambda: (  # type: ignore[method-assign]
        False,
        "No drift detected",
        {"hit_rate": 0.87},
    )
    agent.continuous_trainer.should_retrain = lambda: False  # type: ignore[method-assign]
    _disable_ingestion(agent)

    result = agent.run_once()
    assert result["drift_detected"] is False
    assert result["retraining_triggered"] is False


def test_cooldown_enforcement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    database_url = _db_url(tmp_path)
    init_database(database_url)
    db = AutonomousDB(database_url)
    _seed_model(db)
    _seed_realized_predictions(db, total=40, correct_count=10)
    _seed_model_artifact(tmp_path)
    monkeypatch.chdir(tmp_path)

    agent = MonitoringAgent(db, "1h", "4h")
    agent.validator.validate_all = lambda: {  # type: ignore[method-assign]
        "validated_count": 0,
        "correct_count": 0,
        "hit_rate": 0.0,
        "errors": [],
    }
    agent.drift_detector.check_drift = lambda: (  # type: ignore[method-assign]
        True,
        "forced drift",
        {"hit_rate": 0.2},
    )
    agent.drift_detector.is_in_cooldown = lambda: True  # type: ignore[method-assign]

    called = {"count": 0}

    def _retrain(**kwargs):
        called["count"] += 1
        return {"status": "completed"}

    agent.continuous_trainer.should_retrain = lambda: False  # type: ignore[method-assign]
    agent.continuous_trainer.retrain = _retrain  # type: ignore[method-assign]
    _disable_ingestion(agent)
    result = agent.run_once()

    assert result["drift_detected"] is True
    assert result["retraining_triggered"] is False
    assert called["count"] == 0


def test_schema_preflight_blocks_incompatible_legacy_schema(tmp_path: Path) -> None:
    database_url = _db_url(tmp_path)
    init_database(database_url)
    _create_legacy_prediction_outcomes(database_url)
    with pytest.raises(SchemaCompatibilityError, match="predicted_price"):
        db = AutonomousDB(database_url, auto_upgrade_schema=False)
        MonitoringAgent(db, "1h", "4h")


def test_monitoring_agent_blocks_startup_without_model_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_url = _db_url(tmp_path)
    init_database(database_url)
    db = AutonomousDB(database_url)
    _seed_model(db)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError, match="xgb.json"):
        MonitoringAgent(db, "1h", "4h")


def test_schema_preflight_allows_upgraded_legacy_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_url = _db_url(tmp_path)
    init_database(database_url)
    _create_legacy_prediction_outcomes(database_url)
    db = AutonomousDB(database_url)
    _seed_model(db)
    _seed_realized_predictions(db, total=10, correct_count=6)
    _seed_model_artifact(tmp_path)
    monkeypatch.chdir(tmp_path)

    agent = MonitoringAgent(db, "1h", "4h")
    agent.validator.validate_all = lambda: {  # type: ignore[method-assign]
        "validated_count": 0,
        "correct_count": 0,
        "hit_rate": 0.0,
        "errors": [],
    }
    agent.drift_detector.check_drift = lambda: (  # type: ignore[method-assign]
        False,
        "No drift detected",
        {"hit_rate": 0.6},
    )
    agent.continuous_trainer.should_retrain = lambda: False  # type: ignore[method-assign]
    _disable_ingestion(agent)

    result = agent.run_once()
    assert result["drift_detected"] is False


def test_monitoring_agent_surfaces_runtime_db_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_url = _db_url(tmp_path)
    init_database(database_url)
    db = AutonomousDB(database_url)
    _seed_model(db)
    _seed_realized_predictions(db, total=10, correct_count=5)
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

    def _raise_runtime_db_error() -> dict[str, object]:
        raise MonitorDatabaseError(
            step="predict.store_prediction",
            detail="Schema incompatible: missing prediction_outcomes(predicted_price)",
            remediation="Run --audit then --upgrade",
            error_class="OperationalError",
            database_url=database_url,
        )

    agent.predictor.predict_latest = _raise_runtime_db_error  # type: ignore[method-assign]

    with pytest.raises(MonitorDatabaseError) as exc_info:
        agent.run_once()

    assert exc_info.value.step == "predict.store_prediction"
    assert "upgrade" in exc_info.value.remediation.lower()
