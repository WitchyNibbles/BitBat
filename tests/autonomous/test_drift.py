from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from bitbat.autonomous.db import AutonomousDB
from bitbat.autonomous.drift import DriftDetector
from bitbat.autonomous.models import init_database


def _db_url(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path / 'drift.db'}"


def test_drift_detector_empty_returns_no_drift(tmp_path: Path) -> None:
    database_url = _db_url(tmp_path)
    init_database(database_url)
    db = AutonomousDB(database_url)

    detector = DriftDetector(db, "1h", "4h")
    drift, reason, metrics = detector.check_drift()

    assert drift is False
    assert "Insufficient" in reason
    assert metrics["realized_predictions"] == 0


def test_drift_detector_detects_hit_rate_degradation(tmp_path: Path) -> None:
    database_url = _db_url(tmp_path)
    init_database(database_url)
    db = AutonomousDB(database_url)

    now = datetime.now(UTC).replace(tzinfo=None)
    with db.session() as session:
        db.store_model_version(
            session=session,
            version="v1",
            freq="1h",
            horizon="4h",
            training_start=now - timedelta(days=30),
            training_end=now,
            training_samples=1000,
            cv_score=0.70,
            features=[],
            hyperparameters={},
            training_metadata={},
            is_active=True,
        )

        for idx in range(40):
            pred = db.store_prediction(
                session=session,
                timestamp_utc=now - timedelta(hours=idx + 10),
                predicted_direction="up",
                p_up=0.7,
                p_down=0.2,
                model_version="v1",
                freq="1h",
                horizon="4h",
            )
            actual_direction = "up" if idx < 10 else "down"
            db.realize_prediction(
                session=session,
                prediction_id=pred.id,
                actual_return=0.01 if actual_direction == "up" else -0.01,
                actual_direction=actual_direction,
            )

    detector = DriftDetector(db, "1h", "4h")
    drift, reason, metrics = detector.check_drift()

    assert drift is True
    assert "Hit-rate degradation" in reason or "Losing streak" in reason
    assert metrics["realized_predictions"] == 40


def test_drift_detector_cooldown(tmp_path: Path) -> None:
    database_url = _db_url(tmp_path)
    init_database(database_url)
    db = AutonomousDB(database_url)

    now = datetime.now(UTC).replace(tzinfo=None)
    with db.session() as session:
        event = db.create_retraining_event(
            session=session,
            trigger_reason="manual",
            trigger_metrics={"note": "cooldown-test"},
            old_model_version=None,
        )
        db.complete_retraining_event(
            session=session,
            event_id=event.id,
            new_model_version="v2",
            cv_improvement=0.03,
            training_duration_seconds=12.0,
        )
        event.started_at = now - timedelta(hours=1)

    detector = DriftDetector(db, "1h", "4h")
    assert detector.is_in_cooldown() is True
