"""Phase-level regression gate for Streamlit width compatibility (Phase 7)."""

from __future__ import annotations

import ast
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from bitbat.gui.widgets import get_latest_prediction, get_recent_events, get_system_status

ROOT = Path(__file__).resolve().parents[2]
STREAMLIT_DIR = ROOT / "streamlit"
PAGES_DIR = STREAMLIT_DIR / "pages"


def _runtime_streamlit_files() -> list[Path]:
    files = [STREAMLIT_DIR / "app.py"]
    files.extend(sorted(PAGES_DIR.glob("*.py")))
    return files


def _iter_streamlit_calls(file_path: Path) -> list[ast.Call]:
    tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
    calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if node.func.value.id == "st":
                calls.append(node)
    return calls


def _keyword_value(call: ast.Call, keyword_name: str) -> ast.AST | None:
    for keyword in call.keywords:
        if keyword.arg == keyword_name:
            return keyword.value
    return None


@pytest.fixture()
def phase7_gui_db(tmp_path: Path) -> Path:
    """Minimal autonomous DB fixture for primary GUI workflow signals."""
    db = tmp_path / "autonomous.db"
    con = sqlite3.connect(str(db))
    now = datetime.now(UTC).replace(tzinfo=None)

    con.executescript(
        f"""
        CREATE TABLE performance_snapshots (
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

        CREATE TABLE prediction_outcomes (
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

        CREATE TABLE system_logs (
            id INTEGER PRIMARY KEY,
            created_at TEXT,
            level TEXT,
            message TEXT
        );

        INSERT INTO performance_snapshots
            (snapshot_time, model_version, freq, horizon, window_days,
             total_predictions, realized_predictions, hit_rate)
        VALUES
            ('{(now - timedelta(minutes=20)).isoformat()}', 'v1.0', '1h', '4h', 30,
             100, 80, 0.67);

        INSERT INTO prediction_outcomes
            (timestamp_utc, prediction_timestamp, predicted_direction,
             predicted_return, predicted_price,
             model_version, freq, horizon, created_at)
        VALUES
            ('{now.isoformat()}', '{now.isoformat()}', 'up',
             0.01, 42000.0,
             'v1.0', '1h', '4h', '{now.isoformat()}');

        INSERT INTO system_logs (created_at, level, message)
        VALUES ('{now.isoformat()}', 'INFO', 'Monitoring cycle complete');
        """
    )

    con.commit()
    con.close()
    return db


def test_phase7_runtime_scope_includes_primary_entrypoints() -> None:
    names = {path.name for path in _runtime_streamlit_files()}

    assert "app.py" in names
    assert "0_Quick_Start.py" in names
    assert "4_🔧_System.py" in names


def test_phase7_runtime_width_contract_has_no_deprecated_keywords_or_booleans() -> None:
    deprecated_offenders: list[str] = []
    boolean_width_offenders: list[str] = []

    for file_path in _runtime_streamlit_files():
        for call in _iter_streamlit_calls(file_path):
            deprecated = _keyword_value(call, "use_container_width")
            width_value = _keyword_value(call, "width")

            if deprecated is not None:
                deprecated_offenders.append(f"{file_path.relative_to(ROOT)}:{call.lineno}")

            if isinstance(width_value, ast.Constant) and isinstance(width_value.value, bool):
                boolean_width_offenders.append(f"{file_path.relative_to(ROOT)}:{call.lineno}")

    assert not deprecated_offenders, f"Deprecated width keyword usage found: {deprecated_offenders}"
    assert not boolean_width_offenders, f"Boolean width usage found: {boolean_width_offenders}"


def test_phase7_primary_gui_workflow_signals_remain_operational(phase7_gui_db: Path) -> None:
    status = get_system_status(phase7_gui_db)
    latest_prediction = get_latest_prediction(phase7_gui_db)
    events = get_recent_events(phase7_gui_db, limit=5)

    assert status["status"] == "active"
    assert latest_prediction is not None
    assert latest_prediction["direction"] == "up"
    assert any("Monitoring cycle complete" in event["message"] for event in events)
