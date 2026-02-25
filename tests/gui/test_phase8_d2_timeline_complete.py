"""Phase 8 D2 release gate: timeline data and rendering behavior."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from bitbat.gui.timeline import (
    apply_timeline_filters,
    build_timeline_figure,
    format_timeline_empty_state,
    get_price_series,
    get_timeline_data,
    summarize_timeline_insights,
    summarize_timeline_status,
)

try:
    import plotly  # noqa: F401

    _has_plotly = True
except ImportError:
    _has_plotly = False

D2_CANONICAL_SUITES = [
    "tests/gui/test_timeline.py",
    "tests/gui/test_complete_gui.py",
    "tests/gui/test_phase5_timeline_complete.py",
    "tests/gui/test_phase6_timeline_ux_complete.py",
    "tests/gui/test_phase9_timeline_readability_complete.py",
    "tests/gui/test_phase10_supported_surface_complete.py",
    "tests/gui/test_phase8_d2_timeline_complete.py",
]


def _seed_phase8_d2_db(db_path: Path) -> None:
    con = sqlite3.connect(str(db_path))
    con.execute(
        """
        CREATE TABLE prediction_outcomes (
            id INTEGER PRIMARY KEY,
            timestamp_utc TEXT,
            predicted_direction TEXT,
            p_up REAL,
            p_down REAL,
            predicted_return REAL,
            predicted_price REAL,
            actual_return REAL,
            actual_direction TEXT,
            correct INTEGER,
            freq TEXT,
            horizon TEXT
        )
        """
    )

    rows = [
        (
            "2024-06-01 00:00:00",
            "up",
            0.78,
            0.15,
            0.010,
            43_200.0,
            0.009,
            "up",
            1,
            "1h",
            "4h",
        ),
        (
            "2024-06-01 01:00:00",
            "down",
            0.18,
            0.74,
            -0.007,
            43_050.0,
            0.004,
            "up",
            0,
            "1h",
            "4h",
        ),
        (
            "2024-06-01 02:00:00",
            "up",
            None,
            None,
            0.005,
            43_260.0,
            None,
            None,
            None,
            "1h",
            "4h",
        ),
        (
            "2024-05-20 00:00:00",
            "flat",
            0.33,
            0.33,
            0.000,
            42_800.0,
            0.000,
            "flat",
            1,
            "1h",
            "4h",
        ),
        (
            "2024-05-20 00:15:00",
            "down",
            0.12,
            0.80,
            -0.011,
            42_760.0,
            -0.010,
            "down",
            1,
            "15m",
            "1h",
        ),
    ]

    con.executemany(
        """
        INSERT INTO prediction_outcomes
            (timestamp_utc, predicted_direction, p_up, p_down,
             predicted_return, predicted_price, actual_return, actual_direction,
             correct, freq, horizon)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    con.commit()
    con.close()


def _seed_sparse_prices(data_dir: Path) -> None:
    prices_dir = data_dir / "raw" / "prices"
    prices_dir.mkdir(parents=True)
    prices = pd.DataFrame({
        "timestamp_utc": pd.to_datetime(["2024-06-01 00:00:00"]),
        "close": [43_180.0],
    })
    prices.to_parquet(prices_dir / "btcusd_yf_1h.parquet")


@pytest.mark.skipif(not _has_plotly, reason="plotly not installed")
def test_phase8_d2_release_gate_end_to_end_overlay_and_filter_semantics(tmp_path: Path) -> None:
    db_path = tmp_path / "autonomous.db"
    _seed_phase8_d2_db(db_path)
    _seed_sparse_prices(tmp_path)

    predictions = get_timeline_data(db_path, "1h", "4h", limit=2000)
    assert len(predictions) == 4

    filtered = apply_timeline_filters(predictions, date_window="7d")
    assert len(filtered) == 3

    status = summarize_timeline_status(filtered)
    assert status == {
        "total": 3,
        "completed": 2,
        "correct": 1,
        "pending": 1,
        "accuracy": pytest.approx(50.0),
    }

    insights = summarize_timeline_insights(filtered)
    assert insights["average_confidence"] == pytest.approx(76.0, abs=0.01)

    prices = get_price_series(tmp_path, "1h", filtered["timestamp_utc"].min())
    fig = build_timeline_figure(filtered, prices, show_overlay=True)

    names = {trace.name for trace in fig.data if trace.name}
    assert "BTC Price" in names
    assert "Predicted Return" in names
    assert "Realized Return" in names
    assert "Mismatch Band" in names

    marker_traces = {
        trace.name: trace for trace in fig.data if getattr(trace, "mode", None) == "markers"
    }
    assert marker_traces["UP - Realized (Correct)"].y[0] == pytest.approx(43_180.0)
    assert marker_traces["DOWN - Realized (Wrong)"].y[0] == pytest.approx(43_050.0)
    assert marker_traces["UP - Pending"].y[0] == pytest.approx(43_260.0)

    assert marker_traces["UP - Realized (Correct)"].customdata[0][0] == "78.00%"
    assert marker_traces["UP - Pending"].customdata[0][0] == "n/a"


def test_phase8_d2_release_gate_no_result_filter_message_is_explicit(tmp_path: Path) -> None:
    db_path = tmp_path / "autonomous.db"
    _seed_phase8_d2_db(db_path)

    predictions = get_timeline_data(db_path, "4h", "24h", limit=2000)
    filtered = apply_timeline_filters(predictions, date_window="24h")

    assert filtered.empty
    message = format_timeline_empty_state("4h", "24h", "24h")
    assert "4h / 24h / 24h" in message
    assert "No timeline events match the current filters" in message


def test_phase8_d2_canonical_suite_files_exist() -> None:
    for rel_path in D2_CANONICAL_SUITES:
        assert Path(rel_path).exists(), f"Missing D2 suite file: {rel_path}"
