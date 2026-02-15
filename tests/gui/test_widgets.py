"""Tests for the GUI widgets module (SESSION 2)."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta  # noqa: F401
from pathlib import Path

import pytest

from bitbat.gui.widgets import (
    db_query,
    get_ingestion_status,
    get_latest_prediction,
    get_recent_events,
    get_system_status,
    minutes_until_next_prediction,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
            (datetime('now', '-1 hour'), 'v1.0', '1h', '4h', 30, 50, 40, 0.65);

        INSERT INTO prediction_outcomes
            (timestamp_utc, prediction_timestamp, predicted_direction,
             p_up, p_down, model_version, freq, horizon, created_at)
        VALUES
            (datetime('now'), datetime('now'), 'up',
             0.72, 0.28, 'v1.0', '1h', '4h', datetime('now'));

        INSERT INTO system_logs (created_at, level, message)
        VALUES (datetime('now'), 'INFO', 'Monitoring cycle complete');
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
        assert "🟢" in info["label"]

    def test_idle_when_old_snapshot(self, tmp_path: Path) -> None:
        db = tmp_path / "old.db"
        con = sqlite3.connect(str(db))
        con.execute(
            "CREATE TABLE performance_snapshots "
            "(snapshot_time TEXT, hit_rate REAL, total_predictions INTEGER)"
        )
        # Insert snapshot from 5 hours ago
        old_time = (datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=5)).isoformat()
        con.execute(
            "INSERT INTO performance_snapshots VALUES (?,?,?)",
            (old_time, 0.6, 100),
        )
        con.commit()
        con.close()

        info = get_system_status(db)
        assert info["status"] == "idle"
        assert "🟡" in info["label"]

    def test_active_when_recent_heartbeat(self, tmp_path: Path) -> None:
        db = tmp_path / "autonomous.db"
        sqlite3.connect(str(db)).close()
        heartbeat = tmp_path / "monitoring_agent_heartbeat.json"
        heartbeat.write_text(
            json.dumps(
                {
                    "status": "ok",
                    "updated_at": datetime.now(UTC).replace(tzinfo=None).isoformat(),
                }
            )
        )

        info = get_system_status(db)
        assert info["status"] == "active"
        assert "🟢" in info["label"]

    def test_idle_when_stale_heartbeat(self, tmp_path: Path) -> None:
        db = tmp_path / "autonomous.db"
        sqlite3.connect(str(db)).close()
        heartbeat = tmp_path / "monitoring_agent_heartbeat.json"
        heartbeat.write_text(
            json.dumps(
                {
                    "status": "ok",
                    "updated_at": (
                        datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=4)
                    ).isoformat(),
                }
            )
        )

        info = get_system_status(db)
        assert info["status"] == "idle"
        assert "🟡" in info["label"]


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
        assert pred["p_up"] == pytest.approx(0.72)
        assert pred["confidence"] == pytest.approx(0.72)
        assert pred["model_version"] == "v1.0"

    def test_confidence_is_max_of_p_up_p_down(self, populated_db: Path) -> None:
        pred = get_latest_prediction(populated_db)
        assert pred is not None
        assert pred["confidence"] == max(pred["p_up"], pred["p_down"])


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
        con.execute("CREATE TABLE system_logs (id INTEGER PRIMARY KEY, created_at TEXT, level TEXT, message TEXT)")
        for i in range(15):
            con.execute("INSERT INTO system_logs (created_at, level, message) VALUES (?,?,?)",
                        (datetime.now(UTC).replace(tzinfo=None).isoformat(), "INFO", f"event {i}"))
        con.commit()
        con.close()

        events = get_recent_events(db, limit=5)
        assert len(events) == 5

    def test_reads_timestamp_column_when_present(self, tmp_path: Path) -> None:
        db = tmp_path / "logs_ts.db"
        con = sqlite3.connect(str(db))
        con.execute(
            "CREATE TABLE system_logs (id INTEGER PRIMARY KEY, timestamp TEXT, level TEXT, message TEXT)"
        )
        con.execute(
            "INSERT INTO system_logs (timestamp, level, message) VALUES (?,?,?)",
            (datetime.now(UTC).replace(tzinfo=None).isoformat(), "INFO", "event with timestamp"),
        )
        con.commit()
        con.close()

        events = get_recent_events(db, limit=5)
        assert len(events) == 1
        assert events[0]["message"] == "event with timestamp"


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
        # Created 45 mins ago with 1h interval → ~15 mins remaining
        created = (datetime.now(UTC).replace(tzinfo=None) - timedelta(minutes=45)).isoformat()
        mins = minutes_until_next_prediction(created, interval_hours=1)
        assert mins is not None
        assert 10 <= mins <= 20

    def test_zero_when_overdue(self) -> None:
        # Created 2 hours ago with 1h interval → overdue, returns 0
        old = (datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=2)).isoformat()
        mins = minutes_until_next_prediction(old, interval_hours=1)
        assert mins == 0

    def test_bad_timestamp_returns_none(self) -> None:
        assert minutes_until_next_prediction("not-a-date") is None
