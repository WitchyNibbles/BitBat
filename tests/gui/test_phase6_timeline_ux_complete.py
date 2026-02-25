"""Phase 6 timeline UX expansion regression gates."""

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
)

try:
    import plotly  # noqa: F401

    _has_plotly = True
except ImportError:
    _has_plotly = False


def _seed_phase6_db(db_path: Path) -> None:
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
            "2024-05-01 00:00:00",
            "up",
            0.76,
            0.15,
            0.011,
            43_100.0,
            0.009,
            "up",
            1,
            "1h",
            "4h",
        ),
        (
            "2024-05-01 01:00:00",
            "down",
            0.2,
            0.72,
            -0.008,
            42_980.0,
            -0.002,
            "down",
            1,
            "1h",
            "4h",
        ),
        (
            "2024-05-01 02:00:00",
            "up",
            None,
            None,
            0.004,
            43_150.0,
            None,
            None,
            None,
            "1h",
            "4h",
        ),
        (
            "2024-04-10 00:00:00",
            "flat",
            0.33,
            0.33,
            0.0,
            42_800.0,
            0.0,
            "flat",
            1,
            "1h",
            "4h",
        ),
        (
            "2024-05-01 00:15:00",
            "down",
            0.1,
            0.82,
            -0.01,
            42_850.0,
            -0.012,
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


def _seed_prices(data_dir: Path) -> None:
    prices_dir = data_dir / "raw" / "prices"
    prices_dir.mkdir(parents=True)
    prices = pd.DataFrame({
        "timestamp_utc": pd.to_datetime(
            [
                "2024-05-01 00:00:00",
                "2024-05-01 01:00:00",
                "2024-05-01 02:00:00",
            ]
        ),
        "close": [43_080.0, 43_000.0, 43_140.0],
    })
    prices.to_parquet(prices_dir / "btcusd_yf_1h.parquet")


@pytest.mark.skipif(not _has_plotly, reason="plotly not installed")
def test_phase6_timeline_ux_end_to_end_overlay_and_filters(tmp_path: Path) -> None:
    db_path = tmp_path / "autonomous.db"
    _seed_phase6_db(db_path)
    _seed_prices(tmp_path)

    base = get_timeline_data(db_path, "1h", "4h", limit=2000)
    assert len(base) == 4

    filtered = apply_timeline_filters(base, date_window="7d")
    assert len(filtered) == 3

    insights = summarize_timeline_insights(filtered)
    assert insights["total"] == 3
    assert insights["completed"] == 2
    assert insights["average_confidence"] == pytest.approx(74.0, abs=0.01)

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
    assert marker_traces["UP - Pending"].customdata[0][0] == "n/a"
    assert marker_traces["UP - Realized (Correct)"].customdata[0][0] == "76.00%"
    assert "Confidence: %{customdata[0]}" in marker_traces["UP - Pending"].hovertemplate


def test_phase6_timeline_ux_no_result_filter_message(tmp_path: Path) -> None:
    db_path = tmp_path / "autonomous.db"
    _seed_phase6_db(db_path)

    base = get_timeline_data(db_path, "4h", "24h", limit=2000)
    filtered = apply_timeline_filters(base, date_window="7d")

    assert filtered.empty
    message = format_timeline_empty_state("4h", "24h", "7d")
    assert "4h / 24h / 7d" in message
    assert "No timeline events match the current filters" in message
