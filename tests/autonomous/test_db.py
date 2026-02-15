from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

try:  # pragma: no cover - dependency guard
    import sqlalchemy  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency
    pytest.skip("sqlalchemy not installed", allow_module_level=True)

from bitbat.autonomous.db import AutonomousDB
from bitbat.autonomous.models import init_database


def _db_url(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path / 'autonomous.db'}"


def test_prediction_store_and_realize_flow(tmp_path: Path) -> None:
    database_url = _db_url(tmp_path)
    init_database(database_url)
    db = AutonomousDB(database_url)

    with db.session() as session:
        pred = db.store_prediction(
            session=session,
            timestamp_utc=datetime(2026, 2, 1, 0, 0, 0),
            predicted_direction="up",
            p_up=0.6,
            p_down=0.3,
            model_version="v1",
            freq="1h",
            horizon="4h",
            features_used={"feat_ret_1": 0.01},
        )
        prediction_id = pred.id

    with db.session() as session:
        unrealized = db.get_unrealized_predictions(
            session=session,
            freq="1h",
            horizon="4h",
        )
        assert len(unrealized) == 1
        db.realize_prediction(
            session=session,
            prediction_id=prediction_id,
            actual_return=0.012,
            actual_direction="up",
        )

    with db.session() as session:
        recent = db.get_recent_predictions(session=session, freq="1h", horizon="4h", days=60)
        assert len(recent) == 1
        assert recent[0].correct is True


def test_model_and_retraining_events_flow(tmp_path: Path) -> None:
    database_url = _db_url(tmp_path)
    init_database(database_url)
    db = AutonomousDB(database_url)

    with db.session() as session:
        db.store_model_version(
            session=session,
            version="v1",
            freq="1h",
            horizon="4h",
            training_start=datetime(2026, 1, 1),
            training_end=datetime(2026, 1, 31),
            training_samples=500,
            cv_score=0.55,
            features=["feat_ret_1"],
            hyperparameters={"max_depth": 4},
            training_metadata={"tau": 0.0015},
            is_active=True,
        )
        active = db.get_active_model(session=session, freq="1h", horizon="4h")
        assert active is not None
        assert active.version == "v1"

        event = db.create_retraining_event(
            session=session,
            trigger_reason="manual",
            trigger_metrics={"hit_rate": 0.4},
            old_model_version="v1",
        )
        db.complete_retraining_event(
            session=session,
            event_id=event.id,
            new_model_version="v2",
            cv_improvement=0.03,
            training_duration_seconds=45.0,
        )
        db.store_performance_snapshot(
            session=session,
            model_version="v2",
            freq="1h",
            horizon="4h",
            window_days=30,
            metrics={"total_predictions": 20, "realized_predictions": 18, "hit_rate": 0.61},
        )
        db.log(
            session=session,
            level="INFO",
            service="validator",
            message="snapshot complete",
            details={"model_version": "v2"},
        )


def test_deactivate_old_models_returns_updated_count(tmp_path: Path) -> None:
    database_url = _db_url(tmp_path)
    init_database(database_url)
    db = AutonomousDB(database_url)

    with db.session() as session:
        db.store_model_version(
            session=session,
            version="v1",
            freq="1h",
            horizon="4h",
            training_start=datetime(2026, 1, 1),
            training_end=datetime(2026, 1, 31),
            training_samples=100,
            cv_score=0.5,
            features=[],
            hyperparameters={},
            training_metadata={},
        )
        updated = db.deactivate_old_models(session=session, freq="1h", horizon="4h")
        assert updated == 1

        active = db.get_active_model(session=session, freq="1h", horizon="4h")
        assert active is None
