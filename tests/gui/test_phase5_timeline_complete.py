"""Phase 5 timeline reliability regression gates."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from bitbat.gui.timeline import (
    build_timeline_figure,
    get_price_series,
    get_timeline_data,
    summarize_timeline_status,
)

try:
    import plotly  # noqa: F401

    _has_plotly = True
except ImportError:
    _has_plotly = False


pytestmark = pytest.mark.integration


def _seed_phase5_db(db_path: Path) -> None:
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
            "2024-03-01 00:00:00",
            "up",
            0.72,
            0.21,
            0.01,
            43_000.0,
            0.009,
            "up",
            1,
            "1h",
            "4h",
        ),
        (
            "2024-03-01 01:00:00",
            "down",
            0.2,
            0.74,
            -0.008,
            42_900.0,
            0.004,
            "up",
            0,
            "1h",
            "4h",
        ),
        (
            "2024-03-01 02:00:00",
            "up",
            None,
            None,
            0.006,
            43_150.0,
            None,
            None,
            None,
            "1h",
            "4h",
        ),
        (
            "2024-03-01 00:15:00",
            "down",
            0.12,
            0.81,
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


def _seed_sparse_prices(data_dir: Path) -> None:
    prices_dir = data_dir / "raw" / "prices"
    prices_dir.mkdir(parents=True)
    prices = pd.DataFrame({
        "timestamp_utc": [pd.Timestamp("2024-03-01 00:00:00")],
        "close": [43_050.0],
    })
    prices.to_parquet(prices_dir / "btcusd_yf_1h.parquet")


@pytest.mark.skipif(not _has_plotly, reason="plotly not installed")
def test_phase5_timeline_reliability_end_to_end(tmp_path: Path) -> None:
    db_path = tmp_path / "autonomous.db"
    _seed_phase5_db(db_path)
    _seed_sparse_prices(tmp_path)

    predictions = get_timeline_data(db_path, "1h", "4h")
    assert len(predictions) == 3
    assert predictions["prediction_status"].tolist() == [
        "realized_correct",
        "realized_wrong",
        "pending",
    ]

    prices = get_price_series(tmp_path, "1h", predictions["timestamp_utc"].min())
    fig = build_timeline_figure(predictions, prices)

    assert len(fig.data) == 4  # 1 price line + 3 markers
    marker_traces = {
        trace.name: trace for trace in fig.data if getattr(trace, "mode", None) == "markers"
    }
    assert marker_traces["UP - Realized (Correct)"].y[0] == pytest.approx(43_050.0)
    assert marker_traces["DOWN - Realized (Wrong)"].y[0] == pytest.approx(42_900.0)
    assert marker_traces["UP - Pending"].y[0] == pytest.approx(43_150.0)

    summary = summarize_timeline_status(predictions)
    assert summary == {
        "total": 3,
        "completed": 2,
        "correct": 1,
        "pending": 1,
        "accuracy": pytest.approx(50.0),
    }


def test_phase5_timeline_reliability_limit_window(tmp_path: Path) -> None:
    db_path = tmp_path / "autonomous.db"
    _seed_phase5_db(db_path)

    windowed = get_timeline_data(db_path, "1h", "4h", limit=2)
    assert len(windowed) == 2
    assert windowed["timestamp_utc"].tolist() == [
        pd.Timestamp("2024-03-01 01:00:00"),
        pd.Timestamp("2024-03-01 02:00:00"),
    ]
    assert windowed["prediction_status"].tolist() == ["realized_wrong", "pending"]
