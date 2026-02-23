"""Tests for the prediction timeline chart module."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bitbat.gui.timeline import get_price_series, get_timeline_data

try:
    import plotly  # noqa: F401

    _has_plotly = True
except ImportError:
    _has_plotly = False


def _create_test_db(db_path: Path) -> None:
    """Create a minimal autonomous DB with sample predictions."""
    import sqlite3

    con = sqlite3.connect(str(db_path))
    con.execute(
        """
        CREATE TABLE prediction_outcomes (
            id INTEGER PRIMARY KEY,
            timestamp_utc TEXT,
            predicted_direction TEXT,
            p_up REAL,
            p_down REAL,
            actual_return REAL,
            actual_direction TEXT,
            correct INTEGER,
            freq TEXT,
            horizon TEXT
        )
        """
    )
    rows = [
        ("2024-01-01 00:00:00", "up", 0.7, 0.2, 0.01, "up", 1, "1h", "4h"),
        ("2024-01-01 01:00:00", "down", 0.3, 0.6, -0.005, "down", 1, "1h", "4h"),
        ("2024-01-01 02:00:00", "up", 0.6, 0.3, -0.002, "down", 0, "1h", "4h"),
        ("2024-01-01 03:00:00", "up", 0.65, 0.25, None, None, None, "1h", "4h"),
    ]
    con.executemany(
        """
        INSERT INTO prediction_outcomes
            (timestamp_utc, predicted_direction, p_up, p_down,
             actual_return, actual_direction, correct, freq, horizon)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    con.commit()
    con.close()


def test_get_timeline_data(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    _create_test_db(db_path)

    df = get_timeline_data(db_path, "1h", "4h")

    assert len(df) == 4
    assert "timestamp_utc" in df.columns
    assert "predicted_direction" in df.columns
    assert "correct" in df.columns
    # Should be sorted by timestamp
    assert df["timestamp_utc"].is_monotonic_increasing


def test_get_timeline_data_empty(tmp_path: Path) -> None:
    db_path = tmp_path / "nonexistent.db"
    df = get_timeline_data(db_path, "1h", "4h")
    assert df.empty


def test_get_timeline_data_filters_by_freq(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    _create_test_db(db_path)

    df = get_timeline_data(db_path, "4h", "24h")  # No matching rows
    assert df.empty


def test_get_price_series(tmp_path: Path) -> None:
    prices_dir = tmp_path / "raw" / "prices"
    prices_dir.mkdir(parents=True)

    prices = pd.DataFrame({
        "timestamp_utc": pd.date_range("2024-01-01", periods=24, freq="h"),
        "close": range(100, 124),
    })
    prices.to_parquet(prices_dir / "btcusd_yf_1h.parquet")

    result = get_price_series(tmp_path, "1h", pd.Timestamp("2024-01-01 12:00"))

    assert not result.empty
    assert "close" in result.columns
    assert len(result) == 12  # 12:00 through 23:00


def test_get_price_series_missing(tmp_path: Path) -> None:
    result = get_price_series(tmp_path, "1h", pd.Timestamp("2024-01-01"))
    assert result.empty


@pytest.mark.skipif(not _has_plotly, reason="plotly not installed")
def test_build_timeline_figure() -> None:
    from bitbat.gui.timeline import build_timeline_figure

    predictions = pd.DataFrame({
        "timestamp_utc": pd.date_range("2024-01-01", periods=3, freq="h"),
        "predicted_direction": ["up", "down", "flat"],
        "p_up": [0.7, 0.3, 0.4],
        "p_down": [0.2, 0.6, 0.3],
        "correct": [True, False, None],
    })

    prices = pd.DataFrame(
        {"close": [100.0, 101.0, 99.5]},
        index=pd.date_range("2024-01-01", periods=3, freq="h"),
    )

    fig = build_timeline_figure(predictions, prices)

    # Should be a plotly Figure
    assert hasattr(fig, "data")
    assert hasattr(fig, "layout")
    # Price line + 3 prediction markers = 4 traces
    assert len(fig.data) == 4


@pytest.mark.skipif(not _has_plotly, reason="plotly not installed")
def test_build_timeline_figure_empty_prices() -> None:
    from bitbat.gui.timeline import build_timeline_figure

    predictions = pd.DataFrame({
        "timestamp_utc": pd.date_range("2024-01-01", periods=2, freq="h"),
        "predicted_direction": ["up", "down"],
        "p_up": [0.7, 0.3],
        "p_down": [0.2, 0.6],
        "correct": [None, None],
    })

    prices = pd.DataFrame(columns=["close"])

    fig = build_timeline_figure(predictions, prices)

    # No price line, no markers (can't place without prices)
    assert len(fig.data) == 0
