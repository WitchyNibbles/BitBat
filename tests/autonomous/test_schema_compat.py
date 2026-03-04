from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

try:  # pragma: no cover - dependency guard
    import sqlalchemy  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency
    pytest.skip("sqlalchemy not installed", allow_module_level=True)

from sqlalchemy import text

from bitbat.autonomous.db import AutonomousDB
from bitbat.autonomous.models import create_database_engine, init_database
from bitbat.autonomous.schema_compat import (
    audit_schema_compatibility,
    format_schema_audit,
    required_columns_for_table,
    upgrade_schema_compatibility,
)

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "init_autonomous_db.py"
REPO_ROOT = Path(__file__).resolve().parents[2]


pytestmark = pytest.mark.integration

def _db_url(tmp_path: Path, filename: str) -> str:
    return f"sqlite:///{tmp_path / filename}"


def _create_legacy_prediction_outcomes(
    database_url: str,
    *,
    with_row: bool = False,
) -> None:
    engine = create_database_engine(database_url)
    init_database(database_url, engine=engine)
    with engine.begin() as conn:
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
        if with_row:
            conn.execute(text(
                """
                INSERT INTO prediction_outcomes (
                    timestamp_utc,
                    prediction_timestamp,
                    predicted_direction,
                    p_up,
                    p_down,
                    p_flat,
                    predicted_return,
                    actual_return,
                    actual_direction,
                    correct,
                    model_version,
                    freq,
                    horizon,
                    features_used,
                    created_at,
                    realized_at
                ) VALUES (
                    '2026-02-24 00:00:00',
                    '2026-02-24 00:00:00',
                    'up',
                    0.6,
                    0.3,
                    0.1,
                    0.005,
                    NULL,
                    NULL,
                    NULL,
                    'legacy-v1',
                    '1h',
                    '4h',
                    NULL,
                    '2026-02-24 00:00:00',
                    NULL
                )
                """
            ))
    engine.dispose()


def _create_legacy_performance_snapshots(
    database_url: str,
    *,
    with_row: bool = False,
) -> None:
    engine = create_database_engine(database_url)
    init_database(database_url, engine=engine)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS performance_snapshots"))
        conn.execute(text(
            """
            CREATE TABLE performance_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_version VARCHAR(64) NOT NULL,
                freq VARCHAR(16) NOT NULL,
                horizon VARCHAR(16) NOT NULL,
                snapshot_time DATETIME NOT NULL,
                window_days INTEGER NOT NULL,
                total_predictions INTEGER NOT NULL,
                realized_predictions INTEGER NOT NULL,
                hit_rate FLOAT,
                sharpe_ratio FLOAT,
                avg_return FLOAT,
                max_drawdown FLOAT,
                win_streak INTEGER,
                lose_streak INTEGER,
                calibration_score FLOAT,
                created_at DATETIME NOT NULL
            )
            """
        ))
        if with_row:
            conn.execute(text(
                """
                INSERT INTO performance_snapshots (
                    model_version,
                    freq,
                    horizon,
                    snapshot_time,
                    window_days,
                    total_predictions,
                    realized_predictions,
                    hit_rate,
                    sharpe_ratio,
                    avg_return,
                    max_drawdown,
                    win_streak,
                    lose_streak,
                    calibration_score,
                    created_at
                ) VALUES (
                    'legacy-v1',
                    '1h',
                    '4h',
                    '2026-02-24 00:00:00',
                    30,
                    10,
                    8,
                    0.5,
                    0.1,
                    0.001,
                    -0.1,
                    2,
                    1,
                    0.7,
                    '2026-02-24 00:00:00'
                )
                """
            ))
    engine.dispose()


def _prediction_columns(database_url: str) -> set[str]:
    engine = create_database_engine(database_url)
    with engine.connect() as conn:
        rows = conn.execute(text("PRAGMA table_info(prediction_outcomes)"))
        columns = {str(row[1]) for row in rows}
    engine.dispose()
    return columns


def test_required_contract_contains_runtime_columns() -> None:
    required = required_columns_for_table("prediction_outcomes")
    assert "predicted_price" in required
    assert "predicted_return" in required
    assert "prediction_timestamp" in required


def test_required_contract_contains_performance_snapshot_runtime_columns() -> None:
    required = required_columns_for_table("performance_snapshots")
    assert "snapshot_time" in required
    assert "realized_predictions" in required
    assert "directional_accuracy" in required
    assert "mae" in required
    assert "rmse" in required


def test_audit_detects_legacy_missing_column(tmp_path: Path) -> None:
    database_url = _db_url(tmp_path, "legacy_missing_predicted_price.db")
    _create_legacy_prediction_outcomes(database_url)

    report = audit_schema_compatibility(database_url=database_url)

    assert report.is_compatible is False
    assert report.can_auto_upgrade is True
    assert report.missing_columns == {"prediction_outcomes": ("predicted_price",)}

    table = next(t for t in report.tables if t.table_name == "prediction_outcomes")
    assert table.addable_missing_columns == ("predicted_price",)
    assert table.blocking_missing_columns == ()


def test_audit_detects_legacy_performance_snapshots_missing_columns(tmp_path: Path) -> None:
    database_url = _db_url(tmp_path, "legacy_missing_perf_cols.db")
    _create_legacy_prediction_outcomes(database_url)
    _create_legacy_performance_snapshots(database_url)

    report = audit_schema_compatibility(database_url=database_url)

    assert report.is_compatible is False
    assert report.can_auto_upgrade is True
    assert report.missing_columns["performance_snapshots"] == (
        "directional_accuracy",
        "mae",
        "rmse",
    )


def test_formatted_audit_is_actionable(tmp_path: Path) -> None:
    database_url = _db_url(tmp_path, "legacy_audit_output.db")
    _create_legacy_prediction_outcomes(database_url)

    output = format_schema_audit(audit_schema_compatibility(database_url=database_url))

    assert "Autonomous schema compatibility audit" in output
    assert "prediction_outcomes: incompatible" in output
    assert "missing: predicted_price" in output


def test_init_script_audit_is_non_destructive(tmp_path: Path) -> None:
    database_url = _db_url(tmp_path, "legacy_script_audit.db")
    _create_legacy_prediction_outcomes(database_url)

    result = subprocess.run(  # noqa: S603
        [sys.executable, str(SCRIPT), "--database-url", database_url, "--audit"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1, result.stdout + result.stderr
    assert "Compatibility status: FAIL" in result.stdout
    assert "missing: predicted_price" in result.stdout
    assert "predicted_price" not in _prediction_columns(database_url)


def test_upgrade_is_idempotent_and_preserves_rows(tmp_path: Path) -> None:
    database_url = _db_url(tmp_path, "legacy_upgrade.db")
    _create_legacy_prediction_outcomes(database_url, with_row=True)

    first = upgrade_schema_compatibility(database_url=database_url)
    second = upgrade_schema_compatibility(database_url=database_url)

    assert first.upgraded is True
    assert first.upgrade_state == "upgraded"
    assert first.operation_count == 1
    assert first.missing_columns_before == 1
    assert first.missing_columns_after == 0
    assert ("prediction_outcomes", "predicted_price") in {
        (action.table_name, action.column_name) for action in first.actions
    }
    assert first.is_compatible is True
    assert second.upgraded is False
    assert second.upgrade_state == "already_compatible"
    assert second.operation_count == 0
    assert second.missing_columns_before == 0
    assert second.missing_columns_after == 0
    assert second.is_compatible is True

    engine = create_database_engine(database_url)
    with engine.connect() as conn:
        row = conn.execute(text(
            "SELECT predicted_direction, predicted_return, model_version "
            "FROM prediction_outcomes LIMIT 1"
        )).one()
    engine.dispose()

    assert row[0] == "up"
    assert pytest.approx(float(row[1]), abs=1e-9) == 0.005
    assert row[2] == "legacy-v1"


def test_upgrade_adds_legacy_performance_snapshot_columns(tmp_path: Path) -> None:
    database_url = _db_url(tmp_path, "legacy_perf_upgrade.db")
    _create_legacy_prediction_outcomes(database_url, with_row=True)
    _create_legacy_performance_snapshots(database_url, with_row=True)

    result = upgrade_schema_compatibility(database_url=database_url)

    assert result.is_compatible is True
    assert ("performance_snapshots", "mae") in {
        (action.table_name, action.column_name) for action in result.actions
    }
    assert ("performance_snapshots", "rmse") in {
        (action.table_name, action.column_name) for action in result.actions
    }
    assert ("performance_snapshots", "directional_accuracy") in {
        (action.table_name, action.column_name) for action in result.actions
    }


def test_upgrade_rebuilds_stale_check_constraint_on_retraining_events(tmp_path: Path) -> None:
    """Regression test: old CHECK constraint on retraining_events rejects 'continuous'."""
    database_url = _db_url(tmp_path, "stale_check_constraint.db")
    engine = create_database_engine(database_url)
    init_database(database_url, engine=engine)

    # Replace retraining_events with old schema missing 'continuous' in CHECK
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS retraining_events"))
        conn.execute(text(
            """
            CREATE TABLE retraining_events (
                id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                trigger_reason VARCHAR(32) NOT NULL,
                trigger_metrics JSON,
                old_model_version VARCHAR(64),
                new_model_version VARCHAR(64),
                cv_improvement FLOAT,
                training_duration_seconds FLOAT,
                status VARCHAR(16) NOT NULL,
                error_message TEXT,
                started_at DATETIME NOT NULL,
                completed_at DATETIME,
                CONSTRAINT ck_trigger_reason CHECK (
                    trigger_reason IN (
                        'drift_detected', 'scheduled', 'manual', 'poor_performance'
                    )
                ),
                CONSTRAINT ck_retraining_status CHECK (
                    status IN ('started', 'completed', 'failed')
                )
            )
            """
        ))
        conn.execute(text(
            "INSERT INTO retraining_events (trigger_reason, status, started_at) "
            "VALUES ('manual', 'completed', '2026-02-01 00:00:00')"
        ))
    engine.dispose()

    # AutonomousDB auto-upgrades on init; this should rebuild the table
    db = AutonomousDB(database_url)

    # Inserting trigger_reason='continuous' must succeed after upgrade
    with db.session() as session:
        event = db.create_retraining_event(
            session=session,
            trigger_reason="continuous",
            trigger_metrics=None,
            old_model_version="unknown",
        )
        assert event.id is not None
        assert event.trigger_reason == "continuous"
        assert event.status == "started"

    # Old data must be preserved
    from bitbat.autonomous.models import RetrainingEvent

    with db.session() as session:
        all_events = session.query(RetrainingEvent).order_by(RetrainingEvent.id).all()
        assert len(all_events) == 2
        assert all_events[0].trigger_reason == "manual"
        assert all_events[1].trigger_reason == "continuous"


def test_autonomous_db_init_applies_upgrade_for_legacy_schema(tmp_path: Path) -> None:
    database_url = _db_url(tmp_path, "legacy_runtime_init.db")
    _create_legacy_prediction_outcomes(database_url, with_row=True)

    first_db = AutonomousDB(database_url)
    assert first_db.schema_compatibility_status["upgrade_state"] == "upgraded"
    assert first_db.schema_compatibility_status["operations_applied"] == 1
    assert first_db.schema_compatibility_status["missing_columns_before"] == 1
    assert first_db.schema_compatibility_status["missing_columns_after"] == 0

    db = AutonomousDB(database_url)
    assert db.schema_compatibility_status["upgrade_state"] == "already_compatible"
    assert db.schema_compatibility_status["operations_applied"] == 0
    assert db.schema_compatibility_status["missing_columns_before"] == 0
    assert db.schema_compatibility_status["missing_columns_after"] == 0

    with db.session() as session:
        rows = db.get_recent_predictions(
            session=session,
            freq="1h",
            horizon="4h",
            days=30,
            realized_only=False,
        )
    assert rows
    assert rows[0].model_version == "legacy-v1"
    assert "predicted_price" in _prediction_columns(database_url)
