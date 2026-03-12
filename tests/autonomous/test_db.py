from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

try:  # pragma: no cover - dependency guard
    import sqlalchemy  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency
    pytest.skip("sqlalchemy not installed", allow_module_level=True)

from bitbat.autonomous.db import AutonomousDB, MonitorDatabaseError, classify_monitor_db_error
from bitbat.autonomous.models import init_database

pytestmark = pytest.mark.integration


def _db_url(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path / 'autonomous.db'}"


def _create_legacy_system_logs(database_url: str) -> None:
    init_database(database_url)
    db = AutonomousDB(database_url)
    with db.engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS system_logs"))
        conn.execute(
            text(
                """
                CREATE TABLE system_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at DATETIME,
                    level TEXT,
                    message TEXT
                )
                """
            )
        )
        conn.execute(
            text(
                """
                INSERT INTO system_logs (created_at, level, message)
                VALUES
                    ('2026-03-10 10:00:00', 'INFO', 'older'),
                    ('2026-03-10 11:00:00', 'WARNING', 'newer')
                """
            )
        )


def _create_legacy_retraining_events(database_url: str) -> None:
    init_database(database_url)
    db = AutonomousDB(database_url)
    with db.engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS retraining_events"))
        conn.execute(
            text(
                """
                CREATE TABLE retraining_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trigger_reason TEXT,
                    status TEXT,
                    started_at DATETIME,
                    new_model_version TEXT
                )
                """
            )
        )
        conn.execute(
            text(
                """
                INSERT INTO retraining_events (
                    trigger_reason,
                    status,
                    started_at,
                    new_model_version
                ) VALUES (
                    'manual',
                    'completed',
                    '2026-03-10 11:00:00',
                    'v2'
                )
                """
            )
        )


def _create_legacy_prediction_rows(database_url: str) -> None:
    init_database(database_url)
    db = AutonomousDB(database_url)
    with db.engine.begin() as conn:
        conn.execute(text("DELETE FROM prediction_outcomes"))
        conn.execute(
            text(
                """
                INSERT INTO prediction_outcomes (
                    timestamp_utc,
                    prediction_timestamp,
                    predicted_direction,
                    p_up,
                    p_down,
                    predicted_return,
                    predicted_price,
                    actual_return,
                    actual_direction,
                    correct,
                    model_version,
                    freq,
                    horizon,
                    created_at
                ) VALUES
                    (
                        '2026-03-10 10:00:00',
                        '2026-03-10 10:00:00',
                        'up',
                        0.7,
                        0.2,
                        NULL,
                        NULL,
                        0.01,
                        'up',
                        1,
                        'v1',
                        '1h',
                        '4h',
                        '2026-03-10 10:00:01'
                    ),
                    (
                        '2026-03-10 11:00:00',
                        '2026-03-10 11:00:00',
                        'down',
                        NULL,
                        NULL,
                        -0.004,
                        91500.0,
                        NULL,
                        NULL,
                        NULL,
                        'v2',
                        '1h',
                        '4h',
                        '2026-03-10 11:00:01'
                    )
                """
            )
        )


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


def test_monitor_status_prediction_counts_are_pair_scoped(tmp_path: Path) -> None:
    database_url = _db_url(tmp_path)
    init_database(database_url)
    db = AutonomousDB(database_url)

    with db.session() as session:
        first_pair = []
        for idx in range(3):
            first_pair.append(
                db.store_prediction(
                    session=session,
                    timestamp_utc=datetime(2026, 2, 1, idx, 0, 0),
                    predicted_direction="up",
                    p_up=0.55,
                    p_down=0.35,
                    model_version="v1",
                    freq="1h",
                    horizon="4h",
                )
            )

        db.realize_prediction(
            session=session,
            prediction_id=first_pair[0].id,
            actual_return=0.01,
            actual_direction="up",
        )

        for idx in range(2):
            pred = db.store_prediction(
                session=session,
                timestamp_utc=datetime(2026, 2, 2, idx, 0, 0),
                predicted_direction="down",
                p_up=0.2,
                p_down=0.7,
                model_version="v2",
                freq="5m",
                horizon="30m",
            )
            db.realize_prediction(
                session=session,
                prediction_id=pred.id,
                actual_return=-0.01,
                actual_direction="down",
            )

        counts = db.get_prediction_counts(session=session, freq="1h", horizon="4h")

    assert counts["total_predictions"] == 3
    assert counts["unrealized_predictions"] == 2
    assert counts["realized_predictions"] == 1


def test_monitor_status_prediction_counts_return_zero_for_empty_pair(tmp_path: Path) -> None:
    database_url = _db_url(tmp_path)
    init_database(database_url)
    db = AutonomousDB(database_url)

    with db.session() as session:
        counts = db.get_prediction_counts(session=session, freq="1h", horizon="4h")

    assert counts["total_predictions"] == 0
    assert counts["unrealized_predictions"] == 0
    assert counts["realized_predictions"] == 0


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


def test_classify_monitor_db_error_surfaces_schema_remediation(tmp_path: Path) -> None:
    database_url = _db_url(tmp_path)
    init_database(database_url)
    db = AutonomousDB(database_url)

    raw_error = OperationalError(
        "SELECT predicted_price FROM prediction_outcomes",
        {},
        sqlite3.OperationalError("no such column: prediction_outcomes.predicted_price"),
    )
    classified = classify_monitor_db_error(
        raw_error,
        step="predict.store_prediction",
        database_url=database_url,
        engine=db.engine,
    )

    assert isinstance(classified, MonitorDatabaseError)
    assert classified.step == "predict.store_prediction"
    assert "schema" in classified.detail.lower() or "no such column" in classified.detail.lower()
    assert "--audit" in classified.remediation
    assert "--upgrade" in classified.remediation


def test_classify_monitor_db_error_surfaces_snapshot_schema_remediation(tmp_path: Path) -> None:
    database_url = _db_url(tmp_path)
    init_database(database_url)
    db = AutonomousDB(database_url)

    raw_error = OperationalError(
        "SELECT directional_accuracy FROM performance_snapshots",
        {},
        sqlite3.OperationalError("no such column: performance_snapshots.directional_accuracy"),
    )
    classified = classify_monitor_db_error(
        raw_error,
        step="monitor.status.query_snapshot",
        database_url=database_url,
        engine=db.engine,
    )

    assert isinstance(classified, MonitorDatabaseError)
    assert classified.step == "monitor.status.query_snapshot"
    assert "performance_snapshots" in classified.detail
    assert "--upgrade" in classified.remediation


def test_list_system_logs_uses_legacy_created_at_ordering(tmp_path: Path) -> None:
    database_url = _db_url(tmp_path)
    _create_legacy_system_logs(database_url)

    db = AutonomousDB(database_url)

    logs = db.list_system_logs(limit=10)

    assert logs["total"] == 2
    assert [entry["message"] for entry in logs["logs"]] == ["newer", "older"]
    assert logs["logs"][0]["timestamp"] == "2026-03-10 11:00:00"
    assert logs["logs"][0]["service"] is None


def test_list_retraining_events_and_snapshots_handle_legacy_optional_columns(
    tmp_path: Path,
) -> None:
    database_url = _db_url(tmp_path)
    _create_legacy_retraining_events(database_url)
    init_database(database_url)
    db = AutonomousDB(database_url)

    with db.session() as session:
        db.store_performance_snapshot(
            session=session,
            model_version="v2",
            freq="1h",
            horizon="4h",
            window_days=30,
            metrics={"total_predictions": 12, "realized_predictions": 10, "hit_rate": 0.6},
        )

    events = db.list_retraining_events(limit=5)
    snapshots = db.list_performance_snapshots(limit=5)

    assert events["total"] == 1
    assert events["events"][0]["trigger_reason"] == "manual"
    assert events["events"][0]["training_duration_seconds"] is None
    assert snapshots["snapshots"][0]["model_version"] == "v2"
    assert snapshots["snapshots"][0]["hit_rate"] == pytest.approx(0.6)


def test_get_prediction_views_support_gui_payload_fallbacks(tmp_path: Path) -> None:
    database_url = _db_url(tmp_path)
    _create_legacy_prediction_rows(database_url)

    db = AutonomousDB(database_url)

    latest_prediction = db.get_latest_prediction_payload()
    timeline_rows = db.get_timeline_prediction_rows(freq="1h", horizon="4h", limit=10)

    assert latest_prediction is not None
    assert latest_prediction["direction"] == "down"
    assert latest_prediction["predicted_return"] == pytest.approx(-0.004)
    assert latest_prediction["predicted_price"] == pytest.approx(91500.0)
    assert latest_prediction["confidence"] is None
    assert latest_prediction["created_at"] == "2026-03-10 11:00:01"
    assert len(timeline_rows) == 2
    assert timeline_rows[0]["timestamp_utc"] == "2026-03-10 11:00:00"
    assert timeline_rows[0]["predicted_direction"] == "down"
    assert timeline_rows[1]["correct"] == 1


def test_list_system_logs_retries_transient_lock_then_opens_circuit(tmp_path: Path) -> None:
    database_url = _db_url(tmp_path)
    init_database(database_url)
    db = AutonomousDB(database_url)

    raw_lock = sqlite3.OperationalError("database is locked")
    locked = OperationalError("SELECT * FROM system_logs", {}, raw_lock)

    def always_locked(*args: object, **kwargs: object) -> object:
        raise locked

    db._run_retryable_read = always_locked  # type: ignore[method-assign]

    with pytest.raises(MonitorDatabaseError) as excinfo:
        db.list_system_logs(limit=5)

    err = excinfo.value
    assert err.step == "system.logs"
    assert "temporarily unavailable" in err.detail.lower()
    assert "circuit" in err.detail.lower()
    assert "retry" in err.remediation.lower()


def test_finalize_retraining_success_is_atomic(tmp_path: Path) -> None:
    database_url = _db_url(tmp_path)
    init_database(database_url)
    db = AutonomousDB(database_url)

    with db.session() as session:
        db.store_model_version(
            session=session,
            version="old-v1",
            freq="1h",
            horizon="4h",
            training_start=datetime(2026, 1, 1),
            training_end=datetime(2026, 1, 31),
            training_samples=100,
            cv_score=0.5,
            features=[],
            hyperparameters={},
            training_metadata={},
            is_active=True,
        )
        db.store_model_version(
            session=session,
            version="new-v2",
            freq="1h",
            horizon="4h",
            training_start=datetime(2026, 2, 1),
            training_end=datetime(2026, 2, 28),
            training_samples=120,
            cv_score=0.6,
            features=[],
            hyperparameters={},
            training_metadata={},
            is_active=False,
        )
        event = db.create_retraining_event(
            session=session,
            trigger_reason="manual",
            trigger_metrics={},
            old_model_version="old-v1",
        )
        event_id = int(event.id)

    db.finalize_retraining_success(
        event_id=event_id,
        new_model_version="new-v2",
        freq="1h",
        horizon="4h",
        cv_improvement=0.1,
        training_duration_seconds=30.0,
    )

    with db.session() as session:
        active = db.get_active_model(session=session, freq="1h", horizon="4h")
        finished = session.get(type(event), event_id)
        old_model = (
            session.query(type(active))
            .filter(type(active).version == "old-v1")
            .one()
        )

    assert active is not None
    assert active.version == "new-v2"
    assert finished is not None
    assert finished.status == "completed"
    assert finished.new_model_version == "new-v2"
    assert old_model.is_active is False


def test_finalize_retraining_success_rolls_back_on_missing_event(tmp_path: Path) -> None:
    database_url = _db_url(tmp_path)
    init_database(database_url)
    db = AutonomousDB(database_url)

    with db.session() as session:
        db.store_model_version(
            session=session,
            version="old-v1",
            freq="1h",
            horizon="4h",
            training_start=datetime(2026, 1, 1),
            training_end=datetime(2026, 1, 31),
            training_samples=100,
            cv_score=0.5,
            features=[],
            hyperparameters={},
            training_metadata={},
            is_active=True,
        )
        db.store_model_version(
            session=session,
            version="new-v2",
            freq="1h",
            horizon="4h",
            training_start=datetime(2026, 2, 1),
            training_end=datetime(2026, 2, 28),
            training_samples=120,
            cv_score=0.6,
            features=[],
            hyperparameters={},
            training_metadata={},
            is_active=False,
        )

    with pytest.raises(ValueError):
        db.finalize_retraining_success(
            event_id=999,
            new_model_version="new-v2",
            freq="1h",
            horizon="4h",
            cv_improvement=0.1,
            training_duration_seconds=30.0,
        )

    with db.session() as session:
        old_model = db.get_active_model(session=session, freq="1h", horizon="4h")
        new_model = (
            session.query(type(old_model))
            .filter(type(old_model).version == "new-v2")
            .one()
        )

    assert old_model is not None
    assert old_model.version == "old-v1"
    assert new_model.is_active is False


def test_finalize_retraining_failure_updates_event_atomically(tmp_path: Path) -> None:
    database_url = _db_url(tmp_path)
    init_database(database_url)
    db = AutonomousDB(database_url)

    with db.session() as session:
        event = db.create_retraining_event(
            session=session,
            trigger_reason="manual",
            trigger_metrics={},
            old_model_version="old-v1",
        )
        event_id = int(event.id)

    db.finalize_retraining_failure(event_id=event_id, error_message="boom")

    with db.session() as session:
        finished = session.get(type(event), event_id)

    assert finished is not None
    assert finished.status == "failed"
    assert finished.error_message == "boom"
