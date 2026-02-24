"""Phase 8 D1 release gate: monitor schema compatibility and runtime stability."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import text

from bitbat.autonomous.agent import MonitoringAgent
from bitbat.autonomous.db import AutonomousDB, MonitorDatabaseError
from bitbat.autonomous.models import init_database
from bitbat.autonomous.schema_compat import SchemaCompatibilityError


def _db_url(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path / 'phase8_d1.db'}"


def _seed_model(db: AutonomousDB) -> None:
    now = datetime.now(UTC).replace(tzinfo=None)
    with db.session() as session:
        db.store_model_version(
            session=session,
            version="v1.0.0",
            freq="1h",
            horizon="4h",
            training_start=now - timedelta(days=30),
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


def test_phase8_d1_schema_preflight_blocks_incompatible_legacy_schema(tmp_path: Path) -> None:
    database_url = _db_url(tmp_path)
    init_database(database_url)
    _create_legacy_prediction_outcomes(database_url)

    with pytest.raises(SchemaCompatibilityError, match="predicted_price"):
        db = AutonomousDB(database_url, auto_upgrade_schema=False)
        MonitoringAgent(db, "1h", "4h")


def test_phase8_d1_upgraded_schema_monitor_cycle_stays_operational(tmp_path: Path) -> None:
    database_url = _db_url(tmp_path)
    init_database(database_url)
    _create_legacy_prediction_outcomes(database_url)

    db = AutonomousDB(database_url)
    _seed_model(db)
    _seed_realized_predictions(db, total=16, correct_count=10)

    agent = MonitoringAgent(db, "1h", "4h")
    _disable_ingestion(agent)
    agent.validator.validate_all = lambda: {  # type: ignore[method-assign]
        "validated_count": 0,
        "correct_count": 0,
        "hit_rate": 0.0,
        "errors": [],
    }
    agent.predictor.predict_latest = lambda: None  # type: ignore[method-assign]
    agent.drift_detector.check_drift = lambda: (  # type: ignore[method-assign]
        False,
        "No drift detected",
        {"hit_rate": 0.62},
    )
    agent.continuous_trainer.should_retrain = lambda: False  # type: ignore[method-assign]

    result = agent.run_once()

    assert result["drift_detected"] is False
    assert result["validation_errors"] == []
    assert int(result["metrics"]["realized_predictions"]) == 16


def test_phase8_d1_runtime_db_failure_remains_actionable(tmp_path: Path) -> None:
    database_url = _db_url(tmp_path)
    init_database(database_url)
    db = AutonomousDB(database_url)
    _seed_model(db)
    _seed_realized_predictions(db, total=12, correct_count=6)

    agent = MonitoringAgent(db, "1h", "4h")
    _disable_ingestion(agent)
    agent.validator.validate_all = lambda: {  # type: ignore[method-assign]
        "validated_count": 0,
        "correct_count": 0,
        "hit_rate": 0.0,
        "errors": [],
    }

    def _raise_runtime_db_error() -> None:
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
