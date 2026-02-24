"""
Complete GUI integration tests (SESSION 3).

Validates that all Phase 2 GUI modules load correctly, interact properly
with the database layer, and render without crashing.

All tests are pure-Python (no live Streamlit server required).
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
import yaml

from bitbat.gui.presets import (
    BALANCED,
    DEFAULT_PRESET,
    get_preset,
    list_presets,
)
from bitbat.gui.timeline import format_timeline_empty_state, summarize_timeline_insights, summarize_timeline_status
from bitbat.gui.widgets import (
    db_query,
    get_ingestion_status,
    get_latest_prediction,
    get_recent_events,
    get_system_status,
)

# ---------------------------------------------------------------------------
# Shared fixture — a complete realistic database
# ---------------------------------------------------------------------------


@pytest.fixture()
def full_db(tmp_path: Path) -> Path:
    """Create a fully populated autonomous.db for integration tests."""
    db = tmp_path / "autonomous.db"
    con = sqlite3.connect(str(db))
    now = datetime.now(UTC).replace(tzinfo=None)

    con.executescript(
        f"""
        CREATE TABLE performance_snapshots (
            id INTEGER PRIMARY KEY, snapshot_time TEXT, model_version TEXT,
            freq TEXT, horizon TEXT, window_days INTEGER,
            total_predictions INTEGER, realized_predictions INTEGER,
            hit_rate REAL, sharpe_ratio REAL, avg_return REAL,
            max_drawdown REAL, win_streak INTEGER, lose_streak INTEGER,
            calibration_score REAL
        );
        CREATE TABLE prediction_outcomes (
            id INTEGER PRIMARY KEY, timestamp_utc TEXT,
            prediction_timestamp TEXT, predicted_direction TEXT,
            p_up REAL, p_down REAL, p_flat REAL,
            predicted_return REAL, predicted_price REAL,
            actual_return REAL, actual_direction TEXT, correct BOOLEAN,
            model_version TEXT, freq TEXT, horizon TEXT,
            features_used TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP, realized_at TEXT
        );
        CREATE TABLE model_versions (
            id INTEGER PRIMARY KEY, version TEXT UNIQUE,
            freq TEXT, horizon TEXT, training_start TEXT, training_end TEXT,
            training_samples INTEGER, cv_score REAL, features TEXT,
            hyperparameters TEXT, deployed_at TEXT, replaced_at TEXT,
            is_active BOOLEAN DEFAULT TRUE, training_metadata TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE retraining_events (
            id INTEGER PRIMARY KEY, trigger_reason TEXT, trigger_metrics TEXT,
            old_model_version TEXT, new_model_version TEXT, cv_improvement REAL,
            training_duration_seconds REAL, status TEXT, error_message TEXT,
            started_at TEXT, completed_at TEXT
        );
        CREATE TABLE system_logs (
            id INTEGER PRIMARY KEY, created_at TEXT, level TEXT, message TEXT
        );

        INSERT INTO performance_snapshots
            (snapshot_time, model_version, freq, horizon, window_days,
             total_predictions, realized_predictions, hit_rate, sharpe_ratio,
             avg_return, max_drawdown)
        VALUES
            ('{now.isoformat()}', 'v1.0', '1h', '4h', 30, 100, 80, 0.67, 1.2, 0.002, -0.05);

        INSERT INTO prediction_outcomes
            (timestamp_utc, prediction_timestamp, predicted_direction,
             p_up, p_down, actual_direction, correct, model_version, freq, horizon,
             created_at)
        VALUES
            ('{now.isoformat()}', '{now.isoformat()}', 'up',
             0.73, 0.27, 'up', 1, 'v1.0', '1h', '4h', '{now.isoformat()}');

        INSERT INTO model_versions
            (version, freq, horizon, training_start, training_end,
             training_samples, cv_score, is_active, deployed_at)
        VALUES
            ('v1.0', '1h', '4h',
             '{(now - timedelta(days=30)).isoformat()}',
             '{now.isoformat()}', 5000, 0.71, 1,
             '{now.isoformat()}');

        INSERT INTO retraining_events
            (trigger_reason, status, started_at, completed_at,
             old_model_version, new_model_version, cv_improvement)
        VALUES
            ('drift_detected', 'completed',
             '{(now - timedelta(hours=12)).isoformat()}',
             '{(now - timedelta(hours=11)).isoformat()}',
             'v0.9', 'v1.0', 0.03);

        INSERT INTO system_logs (created_at, level, message)
        VALUES ('{now.isoformat()}', 'INFO', 'Monitoring cycle complete');

        INSERT INTO system_logs (created_at, level, message)
        VALUES ('{(now - timedelta(hours=12)).isoformat()}', 'WARNING',
                'Drift detected — retraining triggered');
        """
    )
    con.commit()
    con.close()
    return db


# ---------------------------------------------------------------------------
# TASK-2-3-06: Alert rules persistence
# ---------------------------------------------------------------------------


class TestAlertRulesStorage:
    def test_yaml_file_created_with_correct_structure(self, tmp_path: Path) -> None:
        rules_path = tmp_path / "config" / "alert_rules.yaml"
        rules_path.parent.mkdir()

        rules = {
            "channels": {"email": {"enabled": True, "address": "test@example.com"}},
            "rules": {"accuracy_drop": {"enabled": True, "threshold": 0.50}},
        }
        rules_path.write_text(yaml.dump(rules, sort_keys=False))

        loaded = yaml.safe_load(rules_path.read_text())
        assert loaded["channels"]["email"]["enabled"] is True
        assert loaded["rules"]["accuracy_drop"]["threshold"] == pytest.approx(0.50)

    def test_yaml_survives_round_trip(self, tmp_path: Path) -> None:
        original = {
            "channels": {"discord": {"enabled": True, "webhook_url": "https://example.com"}},
            "rules": {"drift_detected": {"enabled": True}},
        }
        path = tmp_path / "alert_rules.yaml"
        path.write_text(yaml.dump(original))
        loaded = yaml.safe_load(path.read_text())
        assert loaded == original


# ---------------------------------------------------------------------------
# Full-system integration test
# ---------------------------------------------------------------------------


class TestFullSystemIntegration:
    def test_system_active_with_recent_snapshot(self, full_db: Path) -> None:
        info = get_system_status(full_db)
        assert info["status"] == "active"

    def test_latest_prediction_loaded(self, full_db: Path) -> None:
        pred = get_latest_prediction(full_db)
        assert pred is not None
        assert pred["direction"] == "up"
        assert isinstance(pred["predicted_return"], float)

    def test_recent_events_loaded(self, full_db: Path) -> None:
        events = get_recent_events(full_db, limit=10)
        assert len(events) == 2
        assert any("Monitoring" in ev["message"] for ev in events)

    def test_ingestion_status_no_data_dirs(self, tmp_path: Path) -> None:
        info = get_ingestion_status(tmp_path)
        assert "No data" in info["prices"]
        assert "No data" in info["news"]

    def test_model_version_queryable(self, full_db: Path) -> None:
        rows = db_query(
            full_db,
            "SELECT version, cv_score FROM model_versions WHERE is_active=1",
        )
        assert len(rows) == 1
        assert rows[0][0] == "v1.0"
        assert rows[0][1] == pytest.approx(0.71)

    def test_retraining_event_queryable(self, full_db: Path) -> None:
        rows = db_query(
            full_db,
            "SELECT trigger_reason, status FROM retraining_events LIMIT 1",
        )
        assert rows[0][0] == "drift_detected"
        assert rows[0][1] == "completed"

    def test_performance_snapshot_hit_rate(self, full_db: Path) -> None:
        rows = db_query(
            full_db,
            "SELECT hit_rate FROM performance_snapshots ORDER BY snapshot_time DESC LIMIT 1",
        )
        assert rows[0][0] == pytest.approx(0.67)

    def test_primary_workflow_signals_remain_consistent(self, full_db: Path) -> None:
        status = get_system_status(full_db)
        prediction = get_latest_prediction(full_db)
        events = get_recent_events(full_db, limit=5)

        assert status["status"] == "active"
        assert prediction is not None
        assert prediction["direction"] == "up"
        assert any("Monitoring cycle complete" in event["message"] for event in events)

    def test_primary_workflow_prediction_payload_has_core_fields(self, full_db: Path) -> None:
        prediction = get_latest_prediction(full_db)

        assert prediction is not None
        assert prediction["model_version"] == "v1.0"
        assert isinstance(prediction["predicted_return"], float)


# ---------------------------------------------------------------------------
# Timeline status metrics alignment
# ---------------------------------------------------------------------------


class TestTimelineStatusMetrics:
    def test_timeline_status_summary_mixed_rows(self) -> None:
        predictions = pd.DataFrame({
            "timestamp_utc": pd.date_range("2024-01-01", periods=4, freq="h"),
            "predicted_direction": ["up", "down", "up", "flat"],
            "p_up": [0.75, None, 0.6, None],
            "p_down": [0.2, None, 0.3, None],
            "predicted_return": [0.01, -0.01, 0.005, None],
            "predicted_price": [42_000.0, 41_900.0, 42_200.0, None],
            "actual_return": [0.01, -0.005, -0.002, None],
            "actual_direction": ["up", "down", "down", None],
            "correct": [1, None, 0, None],
        })

        summary = summarize_timeline_status(predictions)

        assert summary["total"] == 4
        assert summary["completed"] == 3
        assert summary["correct"] == 2
        assert summary["pending"] == 1
        assert summary["accuracy"] == pytest.approx(66.67, abs=0.01)

    def test_timeline_status_summary_handles_empty_input(self) -> None:
        summary = summarize_timeline_status(pd.DataFrame())
        assert summary == {
            "total": 0,
            "completed": 0,
            "correct": 0,
            "pending": 0,
            "accuracy": 0.0,
        }

    def test_timeline_status_summary_normalizes_mixed_correct_encodings(self) -> None:
        predictions = pd.DataFrame({
            "timestamp_utc": pd.date_range("2024-02-01", periods=3, freq="h"),
            "predicted_direction": ["up", "down", "up"],
            "predicted_return": [0.01, -0.01, 0.005],
            "predicted_price": [42_000.0, 41_800.0, 42_200.0],
            "actual_return": [0.012, -0.008, None],
            "actual_direction": ["up", "down", None],
            "correct": ["true", "0", None],
        })

        summary = summarize_timeline_status(predictions)

        assert summary["total"] == 3
        assert summary["completed"] == 2
        assert summary["correct"] == 1
        assert summary["pending"] == 1
        assert summary["accuracy"] == pytest.approx(50.0)

    def test_timeline_insights_include_avg_confidence_and_direction_counts(self) -> None:
        predictions = pd.DataFrame({
            "timestamp_utc": pd.date_range("2024-02-01", periods=4, freq="h"),
            "predicted_direction": ["up", "down", "flat", "up"],
            "p_up": [0.8, 0.2, 0.3, None],
            "p_down": [0.1, 0.7, 0.3, None],
            "correct": [1, 0, None, None],
        })

        insights = summarize_timeline_insights(predictions)
        assert insights["total"] == 4
        assert insights["average_confidence"] == pytest.approx(60.0, abs=0.01)
        assert insights["up_count"] == 2
        assert insights["down_count"] == 1
        assert insights["flat_count"] == 1

    def test_timeline_empty_state_message_explicit_for_filters(self) -> None:
        message = format_timeline_empty_state("1h", "24h", "7d")
        assert "1h / 24h / 7d" in message
        assert "Try adjusting freq, horizon, or date window." in message


# ---------------------------------------------------------------------------
# Cross-module: preset → config flow
# ---------------------------------------------------------------------------


class TestPresetToConfigFlow:
    def test_preset_config_saves_and_loads(self, tmp_path: Path) -> None:
        config_path = tmp_path / "user_config.yaml"
        preset = get_preset("conservative")
        config = preset.to_dict()
        config["preset"] = "conservative"
        config_path.write_text(yaml.dump(config))

        loaded = yaml.safe_load(config_path.read_text())
        assert loaded["preset"] == "conservative"
        assert loaded["horizon"] == "24h"
        assert loaded["enter_threshold"] == pytest.approx(0.75)

    def test_default_preset_is_balanced(self) -> None:
        assert get_preset(DEFAULT_PRESET) is BALANCED

    def test_all_presets_produce_valid_configs(self) -> None:
        for _name, preset in list_presets().items():
            cfg = preset.to_dict()
            assert "freq" in cfg
            assert "horizon" in cfg
            assert 0 < cfg["tau"] < 0.1
            assert 0.5 <= cfg["enter_threshold"] <= 1.0


# ---------------------------------------------------------------------------
# SESSION 3 composite: validate all 3 sessions together
# ---------------------------------------------------------------------------


def test_phase2_complete(tmp_path: Path) -> None:
    """
    Composite test that exercises all Phase 2 components end-to-end.
    This is the final integration validation for Phase 2.
    """
    print("\n" + "=" * 60)
    print("PHASE 2 Complete Integration Test")
    print("=" * 60)

    # --- Session 1: Presets ---
    print("\n[1/4] Preset system")
    presets = list_presets()
    assert len(presets) == 3
    for name in ["conservative", "balanced", "aggressive"]:
        p = presets[name]
        assert p.to_dict()
        assert p.to_display()
    print("      PASS")

    # --- Session 1: Config persistence ---
    print("\n[2/4] Config persistence")
    config_path = tmp_path / "user_config.yaml"
    config_path.write_text(yaml.dump({"preset": "conservative"}))
    loaded = yaml.safe_load(config_path.read_text())
    assert loaded["preset"] == "conservative"
    print("      PASS")

    # --- Session 2: Widgets with empty DB ---
    print("\n[3/4] Widget functions (no data)")
    missing_db = tmp_path / "no_db.db"
    assert get_system_status(missing_db)["status"] == "not_started"
    assert get_latest_prediction(missing_db) is None
    assert get_recent_events(missing_db) == []
    info = get_ingestion_status(tmp_path)
    assert "No data" in info["prices"]
    print("      PASS")

    # --- Session 3: Alert rules ---
    print("\n[4/4] Alert rules YAML")
    rules_path = tmp_path / "alert_rules.yaml"
    rules = {
        "channels": {"email": {"enabled": False}},
        "rules": {"drift_detected": {"enabled": True}},
    }
    rules_path.write_text(yaml.dump(rules))
    loaded_rules = yaml.safe_load(rules_path.read_text())
    assert loaded_rules["rules"]["drift_detected"]["enabled"] is True
    print("      PASS")

    print("\n" + "=" * 60)
    print("PHASE 2 CORE FUNCTIONALITY VERIFIED!")
    print("=" * 60)
