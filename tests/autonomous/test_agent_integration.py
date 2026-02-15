from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from bitbat.autonomous.agent import MonitoringAgent
from bitbat.autonomous.db import AutonomousDB
from bitbat.autonomous.models import PerformanceSnapshot, init_database


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


def test_monitoring_agent_integration(tmp_path: Path) -> None:
    database_url = _db_url(tmp_path)
    init_database(database_url)
    db = AutonomousDB(database_url)
    _seed_model(db)
    _seed_realized_predictions(db, total=40, correct_count=15)

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
    agent.retrainer.retrain = lambda **_: {  # type: ignore[method-assign]
        "status": "completed",
        "new_model_version": "v2.0.0",
        "deployed": True,
    }

    result = agent.run_once()

    assert result["drift_detected"] is True
    assert result["retraining_triggered"] is True
    with db.session() as session:
        snapshot_count = session.query(PerformanceSnapshot).count()
    assert snapshot_count > 0


def test_no_drift_scenario(tmp_path: Path) -> None:
    database_url = _db_url(tmp_path)
    init_database(database_url)
    db = AutonomousDB(database_url)
    _seed_model(db)
    _seed_realized_predictions(db, total=40, correct_count=35)

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

    result = agent.run_once()
    assert result["drift_detected"] is False
    assert result["retraining_triggered"] is False


def test_cooldown_enforcement(tmp_path: Path) -> None:
    database_url = _db_url(tmp_path)
    init_database(database_url)
    db = AutonomousDB(database_url)
    _seed_model(db)
    _seed_realized_predictions(db, total=40, correct_count=10)

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

    agent.retrainer.retrain = _retrain  # type: ignore[method-assign]
    result = agent.run_once()

    assert result["drift_detected"] is True
    assert result["retraining_triggered"] is False
    assert called["count"] == 0
