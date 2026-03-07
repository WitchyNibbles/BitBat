"""Phase 9 completion gate: timeline readability and comparison clarity."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from bitbat.gui.timeline import (
    apply_timeline_filters,
    build_timeline_comparison_figure,
    build_timeline_figure,
    get_price_series,
    get_timeline_data,
)

try:
    import plotly  # noqa: F401

    _has_plotly = True
except ImportError:
    _has_plotly = False


pytestmark = pytest.mark.integration

def _seed_phase9_db(db_path: Path) -> None:
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
        ("2024-07-01 00:00:00", "up", 0.79, 0.14, 0.010, 43_300.0, 0.009, "up", 1, "1h", "4h"),
        (
            "2024-07-01 01:00:00",
            "down",
            0.17,
            0.75,
            -0.007,
            43_120.0,
            0.004,
            "up",
            0,
            "1h",
            "4h",
        ),
        ("2024-07-01 02:00:00", "up", 0.72, 0.20, 0.006, 43_250.0, None, None, None, "1h", "4h"),
        (
            "2024-07-01 03:00:00",
            "flat",
            0.33,
            0.33,
            0.000,
            43_210.0,
            None,
            None,
            None,
            "1h",
            "4h",
        ),
        ("2024-07-01 04:00:00", "down", 0.12, 0.81, -0.011, 43_010.0, -0.012, "down", 1, "1h", "4h"),  # noqa: E501
        ("2024-06-10 00:00:00", "up", 0.70, 0.22, 0.005, 42_600.0, 0.004, "up", 1, "1h", "4h"),
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


def _seed_phase9_prices(data_dir: Path) -> None:
    prices_dir = data_dir / "raw" / "prices"
    prices_dir.mkdir(parents=True)
    prices = pd.DataFrame({
        "timestamp_utc": pd.to_datetime(
            [
                "2024-07-01 00:00:00",
                "2024-07-01 01:00:00",
                "2024-07-01 02:00:00",
                "2024-07-01 03:00:00",
                "2024-07-01 04:00:00",
            ]
        ),
        "close": [43_280.0, 43_110.0, 43_240.0, 43_200.0, 43_030.0],
    })
    prices.to_parquet(prices_dir / "btcusd_yf_1h.parquet")


@pytest.mark.skipif(not _has_plotly, reason="plotly not installed")
def test_phase9_readability_gate_default_timeline_and_opt_in_comparison(tmp_path: Path) -> None:
    db_path = tmp_path / "autonomous.db"
    _seed_phase9_db(db_path)
    _seed_phase9_prices(tmp_path)

    predictions = get_timeline_data(db_path, "1h", "4h", limit=2000)
    assert len(predictions) == 6

    filtered = apply_timeline_filters(predictions, date_window="7d")
    assert len(filtered) == 5

    prices = get_price_series(tmp_path, "1h", filtered["timestamp_utc"].min())
    base_fig = build_timeline_figure(filtered, prices, show_overlay=False)

    names = {trace.name for trace in base_fig.data if trace.name}
    assert "BTC Price" in names
    assert "Predicted Return" not in names
    assert "Realized Return" not in names
    assert "Mismatch Band" not in names

    marker_traces = [trace for trace in base_fig.data if getattr(trace, "mode", None) == "markers"]
    assert len(marker_traces) <= 9
    assert sum(len(trace.x) for trace in marker_traces) == len(filtered)

    comparison_fig = build_timeline_comparison_figure(filtered)
    comparison_names = {trace.name for trace in comparison_fig.data if trace.name}
    assert "Predicted Return" in comparison_names
    assert "Realized Return" in comparison_names
    assert "Mismatch Band" in comparison_names

    realized = next(trace for trace in comparison_fig.data if trace.name == "Realized Return")
    assert sum(1 for value in realized.y if pd.isna(value)) >= 2
