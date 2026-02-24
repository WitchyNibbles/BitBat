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
    """Create an autonomous DB with mixed legacy and new prediction rows."""
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
        ("2024-01-01 00:00:00", "up", 0.7, 0.2, None, None, 0.01, "up", 1, "1h", "4h"),
        (
            "2024-01-01 01:00:00",
            "down",
            None,
            None,
            -0.01,
            42_000.0,
            -0.005,
            "down",
            None,
            "1h",
            "4h",
        ),
        ("2024-01-01 02:00:00", "up", 0.55, 0.35, 0.004, 43_000.0, -0.002, "down", 0, "1h", "4h"),
        ("2024-01-01 03:00:00", "up", None, None, 0.006, 44_000.0, None, None, None, "1h", "4h"),
        ("2024-01-01 04:00:00", "down", 0.2, 0.7, -0.02, 40_000.0, -0.02, "down", 1, "4h", "24h"),
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


def _create_matrix_db(db_path: Path) -> None:
    """Create a db matrix spanning windows and freq/horizon combinations."""
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

    rows = []
    base = pd.Timestamp("2024-01-01 00:00:00")
    for i in range(6):
        ts = base + pd.Timedelta(hours=i)
        rows.append(
            (
                str(ts),
                "up" if i % 2 == 0 else "down",
                0.7 if i % 2 == 0 else 0.2,
                0.2 if i % 2 == 0 else 0.7,
                0.005 if i % 2 == 0 else -0.005,
                40_000.0 + i * 100,
                0.003 if i < 3 else None,
                "up" if i < 3 else None,
                1 if i < 3 else None,
                "1h",
                "4h",
            )
        )

    for i in range(2):
        ts = base + pd.Timedelta(minutes=15 * i)
        rows.append(
            (
                str(ts),
                "down",
                0.1,
                0.8,
                -0.01,
                39_500.0 - i * 50,
                -0.008,
                "down",
                1,
                "15m",
                "1h",
            )
        )

    con.executemany(
        """
        INSERT INTO prediction_outcomes
            (timestamp_utc, predicted_direction, p_up, p_down,
             predicted_return, predicted_price, actual_return, actual_direction,
             correct, freq, horizon)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        list(reversed(rows)),
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
    expected_columns = {
        "timestamp_utc",
        "predicted_direction",
        "p_up",
        "p_down",
        "predicted_return",
        "predicted_price",
        "actual_return",
        "actual_direction",
        "correct",
        "confidence",
        "is_realized",
        "prediction_status",
    }
    assert expected_columns.issubset(df.columns)
    # Should be sorted by timestamp
    assert df["timestamp_utc"].is_monotonic_increasing
    assert df["prediction_status"].tolist() == [
        "realized_correct",
        "realized_correct",
        "realized_wrong",
        "pending",
    ]
    assert df["is_realized"].tolist() == [True, True, True, False]
    assert pd.isna(df.loc[1, "confidence"])


def test_get_timeline_data_with_legacy_probability_schema(tmp_path: Path) -> None:
    import sqlite3

    db_path = tmp_path / "legacy.db"
    con = sqlite3.connect(str(db_path))
    con.execute(
        """
        CREATE TABLE prediction_outcomes (
            id INTEGER PRIMARY KEY,
            timestamp_utc TEXT,
            predicted_direction TEXT,
            p_up REAL,
            p_down REAL,
            freq TEXT,
            horizon TEXT
        )
        """
    )
    con.execute(
        """
        INSERT INTO prediction_outcomes
            (timestamp_utc, predicted_direction, p_up, p_down, freq, horizon)
        VALUES ('2024-01-01 00:00:00', 'up', 0.8, 0.1, '1h', '4h')
        """
    )
    con.commit()
    con.close()

    df = get_timeline_data(db_path, "1h", "4h")

    assert len(df) == 1
    assert "predicted_return" in df.columns
    assert pd.isna(df.loc[0, "predicted_return"])
    assert float(df.loc[0, "confidence"]) == pytest.approx(0.8)


def test_get_timeline_data_falls_back_to_prediction_timestamp(tmp_path: Path) -> None:
    import sqlite3

    db_path = tmp_path / "fallback_timestamp.db"
    con = sqlite3.connect(str(db_path))
    con.execute(
        """
        CREATE TABLE prediction_outcomes (
            id INTEGER PRIMARY KEY,
            prediction_timestamp TEXT,
            predicted_direction TEXT,
            p_up REAL,
            p_down REAL,
            freq TEXT,
            horizon TEXT
        )
        """
    )
    con.execute(
        """
        INSERT INTO prediction_outcomes
            (prediction_timestamp, predicted_direction, p_up, p_down, freq, horizon)
        VALUES ('2024-01-01 05:00:00', 'down', 0.2, 0.7, '1h', '4h')
        """
    )
    con.commit()
    con.close()

    df = get_timeline_data(db_path, "1h", "4h")

    assert len(df) == 1
    assert df.loc[0, "timestamp_utc"] == pd.Timestamp("2024-01-01 05:00:00")


def test_get_timeline_data_empty(tmp_path: Path) -> None:
    db_path = tmp_path / "nonexistent.db"
    df = get_timeline_data(db_path, "1h", "4h")
    assert df.empty


def test_get_timeline_data_filters_by_freq(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    _create_test_db(db_path)

    df = get_timeline_data(db_path, "15m", "1h")  # No matching rows
    assert df.empty


def test_get_timeline_data_respects_limit_and_recent_window(tmp_path: Path) -> None:
    db_path = tmp_path / "matrix.db"
    _create_matrix_db(db_path)

    df = get_timeline_data(db_path, "1h", "4h", limit=3)

    assert len(df) == 3
    assert df["timestamp_utc"].tolist() == [
        pd.Timestamp("2024-01-01 03:00:00"),
        pd.Timestamp("2024-01-01 04:00:00"),
        pd.Timestamp("2024-01-01 05:00:00"),
    ]
    assert df["prediction_status"].tolist() == [
        "pending",
        "pending",
        "pending",
    ]


def test_get_timeline_data_routes_by_freq_horizon_pair(tmp_path: Path) -> None:
    db_path = tmp_path / "matrix.db"
    _create_matrix_db(db_path)

    one_hour = get_timeline_data(db_path, "1h", "4h")
    fifteen_min = get_timeline_data(db_path, "15m", "1h")

    assert len(one_hour) == 6
    assert len(fifteen_min) == 2
    assert one_hour["timestamp_utc"].max() > fifteen_min["timestamp_utc"].max()
    assert set(fifteen_min["prediction_status"]) == {"realized_correct"}


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


@pytest.mark.skipif(not _has_plotly, reason="plotly not installed")
def test_build_timeline_figure_status_marker_semantics() -> None:
    from bitbat.gui.timeline import build_timeline_figure

    predictions = pd.DataFrame({
        "timestamp_utc": pd.date_range("2024-01-01", periods=3, freq="h"),
        "predicted_direction": ["up", "down", "up"],
        "correct": [True, False, None],
    })
    prices = pd.DataFrame(
        {"close": [100.0, 99.0, 101.0]},
        index=pd.date_range("2024-01-01", periods=3, freq="h"),
    )

    fig = build_timeline_figure(predictions, prices)
    marker_traces = fig.data[1:]

    assert len(marker_traces) == 3
    assert [trace.marker.opacity for trace in marker_traces] == [1.0, 0.4, 0.75]
    assert [trace.marker.size for trace in marker_traces] == [14, 12, 10]
    assert "Status: Realized (Correct)" in marker_traces[0].hovertemplate
    assert "Status: Realized (Wrong)" in marker_traces[1].hovertemplate
    assert "Status: Pending" in marker_traces[2].hovertemplate


@pytest.mark.skipif(not _has_plotly, reason="plotly not installed")
def test_build_timeline_figure_uses_predicted_price_fallback_for_sparse_prices() -> None:
    from bitbat.gui.timeline import build_timeline_figure

    timestamps = pd.date_range("2024-01-01", periods=3, freq="h")
    predictions = pd.DataFrame({
        "timestamp_utc": timestamps,
        "predicted_direction": ["up", "down", "flat"],
        "predicted_price": [40_000.0, 40_250.0, None],
        "correct": [None, None, None],
    })
    prices = pd.DataFrame(
        {"close": [39_900.0]},
        index=pd.DatetimeIndex([timestamps[0]]),
    )

    fig = build_timeline_figure(predictions, prices)
    marker_traces = fig.data[1:]

    assert len(marker_traces) == 2
    assert marker_traces[0].y[0] == pytest.approx(39_900.0)
    assert marker_traces[1].y[0] == pytest.approx(40_250.0)
