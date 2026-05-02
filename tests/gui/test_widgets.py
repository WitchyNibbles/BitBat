"""Tests for the GUI widgets module."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta  # noqa: F401
from pathlib import Path

import pytest

from bitbat.gui.widgets import (
    cadence_minutes,
    db_query,
    format_relative_time,
    get_cycle_health,
    get_ingestion_status,
    get_latest_prediction,
    get_recent_events,
    get_runtime_summary,
    get_system_status,
    minutes_until_next_prediction,
    normalize_cycle_health,
    normalize_runtime_summary,
    sanitize_heartbeat_payload,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


pytestmark = pytest.mark.integration


@pytest.fixture()
def empty_db(tmp_path: Path) -> Path:
    """DB file that exists but has no tables."""
    db = tmp_path / "empty.db"
    con = sqlite3.connect(str(db))
    con.close()
    return db


@pytest.fixture()
def populated_db(tmp_path: Path) -> Path:
    """DB with minimal schema and one row per table."""
    db = tmp_path / "test.db"
    con = sqlite3.connect(str(db))
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS performance_snapshots (
            id INTEGER PRIMARY KEY,
            snapshot_time TEXT,
            model_version TEXT,
            freq TEXT,
            horizon TEXT,
            window_days INTEGER,
            total_predictions INTEGER,
            realized_predictions INTEGER,
            hit_rate REAL,
            sharpe_ratio REAL,
            avg_return REAL,
            max_drawdown REAL,
            win_streak INTEGER,
            lose_streak INTEGER,
            calibration_score REAL
        );

        CREATE TABLE IF NOT EXISTS prediction_outcomes (
            id INTEGER PRIMARY KEY,
            timestamp_utc TEXT,
            prediction_timestamp TEXT,
            predicted_direction TEXT,
            p_up REAL,
            p_down REAL,
            p_flat REAL,
            predicted_return REAL,
            predicted_price REAL,
            actual_return REAL,
            actual_direction TEXT,
            correct BOOLEAN,
            model_version TEXT,
            freq TEXT,
            horizon TEXT,
            features_used TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            realized_at TEXT
        );

        CREATE TABLE IF NOT EXISTS system_logs (
            id INTEGER PRIMARY KEY,
            created_at TEXT,
            level TEXT,
            message TEXT
        );

        INSERT INTO performance_snapshots
            (snapshot_time, model_version, freq, horizon, window_days,
             total_predictions, realized_predictions, hit_rate)
        VALUES
            (datetime('now', '-1 hour'), 'v1.0', '5m', '30m', 30,
             50, 40, 0.65);

        INSERT INTO prediction_outcomes
            (timestamp_utc, prediction_timestamp, predicted_direction,
             p_up, p_down,
             predicted_return, predicted_price,
             model_version, freq, horizon, created_at)
        VALUES
            (datetime('now'), datetime('now'), 'up',
             0.73, 0.27,
             0.003, 97500.0,
             'v1.0', '5m', '30m', datetime('now'));

        INSERT INTO system_logs (created_at, level, message)
        VALUES (datetime('now'), 'INFO', 'Monitoring cycle complete');
        """
    )
    con.commit()
    con.close()
    return db


@pytest.fixture()
def legacy_prediction_db(tmp_path: Path) -> Path:
    """DB with legacy prediction schema missing confidence/probability columns."""
    db = tmp_path / "legacy_prediction.db"
    con = sqlite3.connect(str(db))
    con.executescript(
        """
        CREATE TABLE prediction_outcomes (
            id INTEGER PRIMARY KEY,
            timestamp_utc TEXT,
            predicted_direction TEXT,
            predicted_return REAL,
            model_version TEXT
        );

        INSERT INTO prediction_outcomes
            (timestamp_utc, predicted_direction, predicted_return, model_version)
        VALUES
            (datetime('now'), 'down', -0.004, 'legacy-v0.9');
        """
    )
    con.commit()
    con.close()
    return db


# ---------------------------------------------------------------------------
# db_query
# ---------------------------------------------------------------------------


class TestDbQuery:
    def test_returns_empty_when_db_missing(self, tmp_path: Path) -> None:
        result = db_query(tmp_path / "nonexistent.db", "SELECT 1")
        assert result == []

    def test_returns_empty_on_bad_query(self, empty_db: Path) -> None:
        result = db_query(empty_db, "SELECT * FROM nonexistent_table")
        assert result == []

    def test_returns_rows_from_valid_query(self, populated_db: Path) -> None:
        rows = db_query(populated_db, "SELECT COUNT(*) FROM prediction_outcomes")
        assert rows == [(1,)]


# ---------------------------------------------------------------------------
# get_system_status
# ---------------------------------------------------------------------------


class TestGetSystemStatus:
    def test_not_started_when_no_db(self, tmp_path: Path) -> None:
        info = get_system_status(tmp_path / "missing.db")
        assert info["status"] == "not_started"

    def test_not_started_when_empty_db(self, empty_db: Path) -> None:
        info = get_system_status(empty_db)
        assert info["status"] == "not_started"

    def test_active_when_recent_snapshot(self, populated_db: Path) -> None:
        info = get_system_status(populated_db)
        assert info["status"] == "active"

    def test_idle_when_old_snapshot(self, tmp_path: Path) -> None:
        db = tmp_path / "old.db"
        con = sqlite3.connect(str(db))
        con.execute(
            "CREATE TABLE performance_snapshots "
            "(snapshot_time TEXT, hit_rate REAL, total_predictions INTEGER)"
        )
        old_time = (datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=5)).isoformat()
        con.execute(
            "INSERT INTO performance_snapshots VALUES (?,?,?)",
            (old_time, 0.6, 100),
        )
        con.commit()
        con.close()

        info = get_system_status(db)
        assert info["status"] == "idle"

    def test_active_when_recent_heartbeat(self, tmp_path: Path) -> None:
        db = tmp_path / "autonomous.db"
        sqlite3.connect(str(db)).close()
        heartbeat = tmp_path / "monitoring_agent_heartbeat.json"
        heartbeat.write_text(
            json.dumps({
                "status": "ok",
                "updated_at": datetime.now(UTC).replace(tzinfo=None).isoformat(),
                "interval_seconds": 300,
            })
        )

        info = get_system_status(db)
        assert info["status"] == "active"

    def test_idle_when_heartbeat_is_stale_for_short_interval(self, tmp_path: Path) -> None:
        db = tmp_path / "autonomous.db"
        sqlite3.connect(str(db)).close()
        heartbeat = tmp_path / "monitoring_agent_heartbeat.json"
        stale_time = datetime.now(UTC).replace(tzinfo=None) - timedelta(minutes=20)
        heartbeat.write_text(
            json.dumps({
                "status": "ok",
                "updated_at": stale_time.isoformat(),
                "interval_seconds": 300,
            })
        )

        info = get_system_status(db)
        assert info["status"] == "idle"

    def test_uses_autonomous_db_activity_summary(self, tmp_path: Path, monkeypatch) -> None:
        db = tmp_path / "autonomous.db"
        sqlite3.connect(str(db)).close()
        monkeypatch.setattr(
            "bitbat.autonomous.db.AutonomousDB.get_system_activity_summary",
            lambda self: {
                "latest_snapshot": datetime.now(UTC).replace(tzinfo=None).isoformat(),
                "latest_monitor_log": None,
                "latest_retraining": None,
            },
        )

        info = get_system_status(db)

        assert info["status"] == "active"


# ---------------------------------------------------------------------------
# get_latest_prediction
# ---------------------------------------------------------------------------


class TestGetLatestPrediction:
    def test_returns_none_when_no_db(self, tmp_path: Path) -> None:
        assert get_latest_prediction(tmp_path / "missing.db") is None

    def test_returns_none_when_empty_db(self, empty_db: Path) -> None:
        assert get_latest_prediction(empty_db) is None

    def test_returns_prediction_dict(self, populated_db: Path) -> None:
        pred = get_latest_prediction(populated_db)
        assert pred is not None
        assert pred["direction"] == "up"
        assert pred["predicted_return"] == pytest.approx(0.003)
        assert pred["predicted_price"] == pytest.approx(97500.0)
        assert pred["model_version"] == "v1.0"

    def test_derives_confidence_from_probability_columns(self, populated_db: Path) -> None:
        pred = get_latest_prediction(populated_db)
        assert pred is not None
        assert pred["p_up"] == pytest.approx(0.73)
        assert pred["p_down"] == pytest.approx(0.27)
        assert pred["confidence"] == pytest.approx(0.73)

    def test_uses_p_flat_for_flat_prediction_confidence(self, tmp_path: Path) -> None:
        db = tmp_path / "flat.db"
        con = sqlite3.connect(str(db))
        con.executescript(
            """
            CREATE TABLE prediction_outcomes (
                id INTEGER PRIMARY KEY,
                timestamp_utc TEXT,
                predicted_direction TEXT,
                p_up REAL,
                p_down REAL,
                p_flat REAL,
                model_version TEXT,
                freq TEXT,
                horizon TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            INSERT INTO prediction_outcomes
                (
                    timestamp_utc,
                    predicted_direction,
                    p_up,
                    p_down,
                    p_flat,
                    model_version,
                    freq,
                    horizon,
                    created_at
                )
            VALUES
                (datetime('now'), 'flat', 0.01, 0.02, 0.97, 'v-flat', '5m', '30m', datetime('now'));
            """
        )
        con.commit()
        con.close()

        pred = get_latest_prediction(db)

        assert pred is not None
        assert pred["direction"] == "flat"
        assert pred["p_flat"] == pytest.approx(0.97)
        assert pred["confidence"] == pytest.approx(0.97)

    def test_handles_legacy_prediction_schema_without_confidence_columns(
        self, legacy_prediction_db: Path
    ) -> None:
        pred = get_latest_prediction(legacy_prediction_db)
        assert pred is not None
        assert pred["direction"] == "down"
        assert pred["predicted_return"] == pytest.approx(-0.004)
        assert pred["predicted_price"] is None
        assert pred["confidence"] is None

    def test_legacy_prediction_payload_keeps_confidence_key_shape(
        self, legacy_prediction_db: Path
    ) -> None:
        pred = get_latest_prediction(legacy_prediction_db)

        assert pred is not None
        assert "confidence" in pred
        assert "created_at" in pred
        assert pred["created_at"] == pred["timestamp_utc"]

    def test_uses_autonomous_db_latest_prediction_payload(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        db = tmp_path / "autonomous.db"
        sqlite3.connect(str(db)).close()
        monkeypatch.setattr(
            "bitbat.autonomous.db.AutonomousDB.get_latest_prediction_payload",
            lambda self: {
                "timestamp_utc": "2026-03-12T12:00:00",
                "direction": "up",
                "predicted_return": 0.01,
                "predicted_price": 100000.0,
                "model_version": "v2",
                "created_at": "2026-03-12T12:00:01",
                "p_up": 0.8,
                "p_down": 0.1,
                "confidence": 0.8,
            },
        )

        pred = get_latest_prediction(db)

        assert pred is not None
        assert pred["model_version"] == "v2"
        assert pred["confidence"] == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# get_recent_events
# ---------------------------------------------------------------------------


class TestGetRecentEvents:
    def test_returns_empty_when_no_db(self, tmp_path: Path) -> None:
        assert get_recent_events(tmp_path / "missing.db") == []

    def test_returns_events_list(self, populated_db: Path) -> None:
        events = get_recent_events(populated_db, limit=5)
        assert len(events) >= 1
        ev = events[0]
        assert "time" in ev
        assert "level" in ev
        assert "message" in ev

    def test_limit_respected(self, tmp_path: Path) -> None:
        db = tmp_path / "logs.db"
        con = sqlite3.connect(str(db))
        con.execute(
            "CREATE TABLE system_logs "
            "(id INTEGER PRIMARY KEY, created_at TEXT, "
            "level TEXT, message TEXT)"
        )
        for i in range(15):
            con.execute(
                "INSERT INTO system_logs " "(created_at, level, message) VALUES (?,?,?)",
                (
                    datetime.now(UTC).replace(tzinfo=None).isoformat(),
                    "INFO",
                    f"event {i}",
                ),
            )
        con.commit()
        con.close()

        events = get_recent_events(db, limit=5)
        assert len(events) == 5

    def test_reads_timestamp_column_when_present(self, tmp_path: Path) -> None:
        db = tmp_path / "logs_ts.db"
        con = sqlite3.connect(str(db))
        con.execute(
            "CREATE TABLE system_logs "
            "(id INTEGER PRIMARY KEY, timestamp TEXT, "
            "level TEXT, message TEXT)"
        )
        con.execute(
            "INSERT INTO system_logs " "(timestamp, level, message) VALUES (?,?,?)",
            (
                datetime.now(UTC).replace(tzinfo=None).isoformat(),
                "INFO",
                "event with timestamp",
            ),
        )
        con.commit()
        con.close()

        events = get_recent_events(db, limit=5)
        assert len(events) == 1
        assert events[0]["message"] == "event with timestamp"

    def test_uses_autonomous_db_recent_event_payload(self, tmp_path: Path, monkeypatch) -> None:
        db = tmp_path / "autonomous.db"
        sqlite3.connect(str(db)).close()
        monkeypatch.setattr(
            "bitbat.autonomous.db.AutonomousDB.list_recent_system_events",
            lambda self, *, limit: [
                {
                    "time": "2026-03-12T12:00:00",
                    "level": "INFO",
                    "message": "Monitor cycle complete",
                }
            ],
        )

        events = get_recent_events(db, limit=5)

        assert len(events) == 1
        assert events[0]["message"] == "Monitor cycle complete"


# ---------------------------------------------------------------------------
# get_ingestion_status
# ---------------------------------------------------------------------------


class TestGetIngestionStatus:
    def test_no_data_when_dir_missing(self, tmp_path: Path) -> None:
        info = get_ingestion_status(tmp_path)
        assert "No data" in info["prices"]
        assert "No data" in info["news"]

    def test_fresh_when_recent_file(self, tmp_path: Path) -> None:
        prices_dir = tmp_path / "raw" / "prices"
        prices_dir.mkdir(parents=True)
        (prices_dir / "data.parquet").write_bytes(b"fake")
        info = get_ingestion_status(tmp_path)
        assert "Fresh" in info["prices"]


# ---------------------------------------------------------------------------
# minutes_until_next_prediction
# ---------------------------------------------------------------------------


class TestMinutesUntilNextPrediction:
    def test_returns_none_when_no_timestamp(self) -> None:
        assert minutes_until_next_prediction(None) is None

    def test_returns_zero_or_positive(self) -> None:
        recent = (datetime.now(UTC).replace(tzinfo=None) - timedelta(minutes=30)).isoformat()
        mins = minutes_until_next_prediction(recent, interval_hours=1)
        assert mins is not None
        assert mins >= 0

    def test_approximately_correct(self) -> None:
        created = (datetime.now(UTC).replace(tzinfo=None) - timedelta(minutes=45)).isoformat()
        mins = minutes_until_next_prediction(created, interval_hours=1)
        assert mins is not None
        assert 10 <= mins <= 20

    def test_zero_when_overdue(self) -> None:
        old = (datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=2)).isoformat()
        mins = minutes_until_next_prediction(old, interval_hours=1)
        assert mins == 0

    def test_bad_timestamp_returns_none(self) -> None:
        assert minutes_until_next_prediction("not-a-date") is None

    def test_supports_minute_based_interval(self) -> None:
        created = (datetime.now(UTC).replace(tzinfo=None) - timedelta(minutes=45)).isoformat()
        mins = minutes_until_next_prediction(created, interval_minutes=90)
        assert mins is not None
        assert 40 <= mins <= 50


class TestCadenceHelpers:
    def test_cadence_minutes_maps_known_frequency(self) -> None:
        assert cadence_minutes("15m") == 15

    def test_cadence_minutes_falls_back_to_hourly(self) -> None:
        assert cadence_minutes("mystery") == 60

    def test_format_relative_time_returns_human_label(self) -> None:
        timestamp = (datetime.now(UTC).replace(tzinfo=None) - timedelta(minutes=5)).isoformat()
        label = format_relative_time(timestamp)
        assert label is not None
        assert label.endswith("ago")


class TestGetRuntimeSummary:
    def test_active_summary_reports_signal_and_next_cycle(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        db = tmp_path / "autonomous.db"
        db.touch()

        monkeypatch.setattr(
            "bitbat.gui.widgets.get_system_status",
            lambda _: {"status": "active", "label": "🟢 Active", "hours_ago": 0.1},
        )
        monkeypatch.setattr(
            "bitbat.gui.widgets.get_latest_prediction",
            lambda _: {
                "direction": "up",
                "confidence": 0.81,
                "created_at": datetime.now(UTC).replace(tzinfo=None).isoformat(),
            },
        )
        monkeypatch.setattr(
            "bitbat.gui.widgets.get_ingestion_status",
            lambda _: {"prices": "🟢 Fresh", "news": "🟡 1h ago"},
        )

        summary = get_runtime_summary(db, tmp_path, interval_minutes=60)

        assert summary["status"] == "active"
        assert summary["title"] == "Watcher Active"
        assert "Next cycle" in summary["next"]
        assert "UP" in summary["detail"]

    def test_warming_summary_handles_active_runtime_without_prediction(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        db = tmp_path / "autonomous.db"
        db.touch()

        monkeypatch.setattr(
            "bitbat.gui.widgets.get_system_status",
            lambda _: {"status": "active", "label": "🟢 Active", "hours_ago": 0.2},
        )
        monkeypatch.setattr("bitbat.gui.widgets.get_latest_prediction", lambda _: None)
        monkeypatch.setattr(
            "bitbat.gui.widgets.get_ingestion_status",
            lambda _: {"prices": "🟢 Fresh", "news": "⚪ No data"},
        )

        summary = get_runtime_summary(db, tmp_path, interval_minutes=60)

        assert summary["status"] == "warming"
        assert summary["title"] == "Watcher Warming"
        assert "first completed prediction cycle" in summary["next"]

    def test_dormant_summary_guides_first_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "bitbat.gui.widgets.get_system_status",
            lambda _: {"status": "not_started", "label": "⚪ Not Started", "hours_ago": None},
        )
        monkeypatch.setattr("bitbat.gui.widgets.get_latest_prediction", lambda _: None)
        monkeypatch.setattr(
            "bitbat.gui.widgets.get_ingestion_status",
            lambda _: {"prices": "⚪ No data", "news": "⚪ No data"},
        )

        summary = get_runtime_summary(tmp_path / "missing.db", tmp_path, interval_minutes=60)

        assert summary["status"] == "dormant"
        assert summary["title"] == "Watcher Dormant"
        assert "Train a model on Quick Start" in summary["next"]

    def test_blocked_summary_surfaces_price_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        db = tmp_path / "autonomous.db"
        db.touch()

        monkeypatch.setattr(
            "bitbat.gui.widgets.get_system_status",
            lambda _: {"status": "active", "label": "🟢 Active", "hours_ago": 0.1},
        )
        monkeypatch.setattr("bitbat.gui.widgets.get_latest_prediction", lambda _: None)
        monkeypatch.setattr(
            "bitbat.gui.widgets.get_ingestion_status",
            lambda _: {"prices": "🔴 stale", "news": "🟢 Fresh"},
        )
        monkeypatch.setattr(
            "bitbat.gui.widgets.get_cycle_health",
            lambda *_args, **_kwargs: {
                "state": "blocked",
                "tone": "danger",
                "title": "Prediction Blocked",
                "summary": "Price candles failed, so the watcher skipped the latest signal.",
                "action": "Restore price candles and let the next cycle retry.",
                "issues": [{"source": "prices", "message": "Price refresh failed"}],
                "issue_count": 1,
            },
        )

        summary = get_runtime_summary(db, tmp_path, interval_minutes=60)

        assert summary["status"] == "blocked"
        assert summary["tone"] == "danger"
        assert summary["title"] == "Prediction Blocked"
        assert "skipped" in summary["detail"].lower()

    def test_active_summary_recovers_when_cycle_health_payload_is_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        db = tmp_path / "autonomous.db"
        db.touch()

        monkeypatch.setattr(
            "bitbat.gui.widgets.get_system_status",
            lambda _: {"status": "active", "label": "🟢 Active", "hours_ago": 0.1},
        )
        monkeypatch.setattr(
            "bitbat.gui.widgets.get_latest_prediction",
            lambda _: {
                "direction": "up",
                "confidence": 0.81,
                "created_at": datetime.now(UTC).replace(tzinfo=None).isoformat(),
            },
        )
        monkeypatch.setattr(
            "bitbat.gui.widgets.get_ingestion_status",
            lambda _: {"prices": "🟢 Fresh", "news": "🟡 1h ago"},
        )
        monkeypatch.setattr("bitbat.gui.widgets.get_cycle_health", lambda *_args, **_kwargs: {})

        summary = get_runtime_summary(db, tmp_path, interval_minutes=60)

        assert summary["status"] == "active"
        assert summary["cycle_health"]["state"] == "unknown"
        assert (
            summary["cycle_health"]["summary"] == "Cycle health details are unavailable right now."
        )


class TestNormalizeRuntimeSummary:
    def test_fills_missing_runtime_and_cycle_health_keys(self) -> None:
        runtime = normalize_runtime_summary({
            "status": "warming",
            "title": "Watcher Warming",
        })

        assert runtime["status"] == "warming"
        assert runtime["tone"] == "info"
        assert runtime["cycle_health"]["state"] == "unknown"
        assert runtime["cycle_health"]["issues"] == []


class TestNormalizeCycleHealth:
    def test_fills_missing_cycle_health_keys(self) -> None:
        cycle_health = normalize_cycle_health({"state": "degraded"})

        assert cycle_health["state"] == "degraded"
        assert cycle_health["summary"] == "Cycle health details are unavailable right now."
        assert cycle_health["issue_count"] == 0
        assert cycle_health["issues"] == []


class TestSanitizeHeartbeatPayload:
    def test_hides_sensitive_heartbeat_fields(self) -> None:
        payload = sanitize_heartbeat_payload({
            "status": "ok",
            "updated_at": "2026-05-02T12:00:00",
            "database_url": "postgresql://user:secret@example/db",
            "config_path": "/workspace/runtime.yaml",
            "error": "traceback details",
            "cycle_prediction_state": "generated",
            "cycle_ingestion_failures": [
                {
                    "source": "prices",
                    "required": True,
                    "status": "failed",
                    "message": "Price refresh failed",
                    "details": {"error": "secret path"},
                }
            ],
        })

        assert payload is not None
        assert "database_url" not in payload
        assert "config_path" not in payload
        assert "error" not in payload
        assert payload["error_summary"] == "traceback details"
        assert payload["sensitive_fields_hidden"] is True
        assert payload["cycle_ingestion_failures"][0] == {
            "source": "prices",
            "required": True,
            "status": "failed",
            "message": "Price refresh failed",
        }

    def test_redacts_paths_and_urls_in_error_summary(self) -> None:
        payload = sanitize_heartbeat_payload({
            "status": "error",
            "error": (
                "database_url=postgresql://user:secret@example/db "
                "config_path=/tmp/runtime.yaml see https://example.test/failure"
            ),
        })

        assert payload is not None
        assert payload["error_summary"] == (
            "database_url=<redacted> config_path=<redacted> see <redacted-url>"
        )


class TestGetCycleHealth:
    def test_hides_raw_error_text_in_monitor_error_state(self, tmp_path: Path) -> None:
        db = tmp_path / "autonomous.db"
        db.touch()
        heartbeat = tmp_path / "monitoring_agent_heartbeat.json"
        heartbeat.write_text(
            json.dumps({
                "status": "error",
                "updated_at": datetime.now(UTC).replace(tzinfo=None).isoformat(),
                "error": "database_url=postgres://secret@example/db config_path=/tmp/x",
            }),
            encoding="utf-8",
        )

        health = get_cycle_health(db, interval_minutes=60)

        assert health["state"] == "error"
        assert health["summary"] == "Latest cycle failed before completion."

    def test_reports_blocking_price_failure_in_latest_cycle(self, tmp_path: Path) -> None:
        db = tmp_path / "autonomous.db"
        db.touch()
        heartbeat = tmp_path / "monitoring_agent_heartbeat.json"
        heartbeat.write_text(
            json.dumps({
                "status": "ok",
                "updated_at": datetime.now(UTC).replace(tzinfo=None).isoformat(),
                "cycle_ingestion_state": "degraded",
                "cycle_ingestion_failures": [
                    {
                        "source": "prices",
                        "required": True,
                        "status": "failed",
                        "message": "Price refresh failed",
                        "details": {"error": "upstream outage"},
                    }
                ],
            }),
            encoding="utf-8",
        )

        health = get_cycle_health(db, interval_minutes=60)

        assert health["state"] == "blocked"
        assert health["tone"] == "danger"
        assert health["issue_count"] == 1
        assert health["issues"][0]["source"] == "prices"

    def test_reports_optional_feed_degradation_in_latest_cycle(self, tmp_path: Path) -> None:
        db = tmp_path / "autonomous.db"
        db.touch()
        heartbeat = tmp_path / "monitoring_agent_heartbeat.json"
        heartbeat.write_text(
            json.dumps({
                "status": "ok",
                "updated_at": datetime.now(UTC).replace(tzinfo=None).isoformat(),
                "cycle_ingestion_state": "degraded",
                "cycle_ingestion_failures": [
                    {
                        "source": "news",
                        "required": False,
                        "status": "failed",
                        "message": "News refresh failed",
                        "details": {"error": "rss timeout"},
                    }
                ],
                "cycle_prediction_state": "generated",
            }),
            encoding="utf-8",
        )

        health = get_cycle_health(db, interval_minutes=60)

        assert health["state"] == "degraded"
        assert health["tone"] == "warning"
        assert health["issue_count"] == 1
        assert health["issues"][0]["source"] == "news"

    def test_reports_stale_heartbeat_against_expected_interval(self, tmp_path: Path) -> None:
        db = tmp_path / "autonomous.db"
        db.touch()
        heartbeat = tmp_path / "monitoring_agent_heartbeat.json"
        stale_time = datetime.now(UTC).replace(tzinfo=None) - timedelta(minutes=20)
        heartbeat.write_text(
            json.dumps({
                "status": "ok",
                "updated_at": stale_time.isoformat(),
                "interval_seconds": 300,
                "cycle_prediction_state": "generated",
            }),
            encoding="utf-8",
        )

        health = get_cycle_health(db, interval_minutes=5)

        assert health["state"] == "degraded"
        assert health["title"] == "Watcher Stale"
        assert "older than the expected watcher cadence" in health["summary"]
