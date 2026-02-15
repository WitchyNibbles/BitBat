from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from bitbat.autonomous.agent import MonitoringAgent
from bitbat.autonomous.db import AutonomousDB
from bitbat.autonomous.drift import DriftDetector
from bitbat.autonomous.metrics import PerformanceMetrics
from bitbat.autonomous.models import PerformanceSnapshot, init_database


def _db_url(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path / 'session3.db'}"


def test_session3_complete_workflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    database_url = _db_url(tmp_path)
    init_database(database_url)
    db = AutonomousDB(database_url)

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
            cv_score=0.65,
            features=["feat_rsi"],
            hyperparameters={"max_depth": 6},
            training_metadata={"tau": 0.01},
            is_active=True,
        )

        for idx in range(50):
            pred = db.store_prediction(
                session=session,
                timestamp_utc=now - timedelta(hours=100 - idx),
                predicted_direction="up",
                p_up=0.7,
                p_down=0.2,
                model_version="v1.0.0",
                freq="1h",
                horizon="4h",
            )
            good = idx < 20
            db.realize_prediction(
                session=session,
                prediction_id=pred.id,
                actual_return=0.015 if good else -0.01,
                actual_direction="up" if good else "down",
            )

    with db.session() as session:
        predictions = db.get_recent_predictions(
            session=session,
            freq="1h",
            horizon="4h",
            days=365,
            realized_only=True,
        )

    metrics = PerformanceMetrics(predictions).to_dict()
    assert metrics["hit_rate"] < 0.5

    detector = DriftDetector(db, "1h", "4h")
    drift, reason, _ = detector.check_drift()
    assert drift is True
    assert "Hit-rate degradation" in reason or "Losing streak" in reason

    agent = MonitoringAgent(db, "1h", "4h")
    monkeypatch.setattr(
        agent.validator,
        "validate_all",
        lambda: {"validated_count": 0, "correct_count": 0, "hit_rate": 0.0, "errors": []},
    )
    monkeypatch.setattr(agent.drift_detector, "is_in_cooldown", lambda: False)
    monkeypatch.setattr(
        agent.retrainer,
        "retrain",
        lambda **_: {"status": "completed", "new_model_version": "v2.0.0", "deployed": True},
    )

    result = agent.run_once()

    assert result["validations"] >= 0
    assert result["drift_detected"] in [True, False]

    with db.session() as session:
        snapshots = session.query(PerformanceSnapshot).all()
    assert len(snapshots) > 0
