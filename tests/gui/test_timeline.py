"""Tests for the prediction timeline chart module."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bitbat.gui.timeline import (
    apply_timeline_filters,
    build_timeline_comparison_figure,
    build_timeline_overlay_frame,
    format_timeline_empty_state,
    get_price_series,
    get_timeline_data,
    list_timeline_filter_options,
    summarize_timeline_insights,
)

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


def test_list_timeline_filter_options_uses_db_pairs(tmp_path: Path) -> None:
    db_path = tmp_path / "options.db"
    _create_test_db(db_path)

    freqs, horizons = list_timeline_filter_options(db_path, "1h", "4h")

    assert freqs == ["1h", "4h"]
    assert horizons == ["4h", "24h"]


def test_apply_timeline_filters_default_last_7_days() -> None:
    predictions = pd.DataFrame({
        "timestamp_utc": pd.to_datetime(
            [
                "2024-01-01 00:00:00",
                "2024-01-08 00:00:00",
                "2024-01-10 00:00:00",
            ]
        ),
        "predicted_direction": ["up", "down", "flat"],
        "correct": [1, None, None],
    })

    filtered = apply_timeline_filters(predictions)
    assert filtered["timestamp_utc"].tolist() == [
        pd.Timestamp("2024-01-08 00:00:00"),
        pd.Timestamp("2024-01-10 00:00:00"),
    ]


def test_apply_timeline_filters_all_keeps_full_set() -> None:
    predictions = pd.DataFrame({
        "timestamp_utc": pd.date_range("2024-01-01", periods=4, freq="d"),
        "predicted_direction": ["up", "down", "up", "flat"],
        "correct": [1, 0, None, None],
    })

    filtered = apply_timeline_filters(predictions, date_window="all")
    assert len(filtered) == 4


def test_summarize_timeline_insights_includes_avg_confidence() -> None:
    predictions = pd.DataFrame({
        "timestamp_utc": pd.date_range("2024-01-01", periods=3, freq="h"),
        "predicted_direction": ["up", "down", "up"],
        "p_up": [0.8, 0.2, None],
        "p_down": [0.1, 0.7, None],
        "correct": [1, 0, None],
    })

    summary = summarize_timeline_insights(predictions)
    assert summary["total"] == 3
    assert summary["average_confidence"] == pytest.approx(75.0)
    assert summary["up_count"] == 2
    assert summary["down_count"] == 1
    assert summary["flat_count"] == 0


def test_summarize_timeline_insights_matches_filtered_subset() -> None:
    predictions = pd.DataFrame({
        "timestamp_utc": pd.to_datetime(
            [
                "2024-01-01 00:00:00",
                "2024-01-08 00:00:00",
                "2024-01-09 00:00:00",
                "2024-01-10 00:00:00",
            ]
        ),
        "predicted_direction": ["up", "down", "up", "flat"],
        "p_up": [0.8, 0.2, 0.7, None],
        "p_down": [0.1, 0.7, 0.2, None],
        "correct": [1, 0, 0, None],
    })

    filtered = apply_timeline_filters(predictions, date_window="24h")
    summary = summarize_timeline_insights(filtered)

    assert len(filtered) == 2
    assert summary["total"] == 2
    assert summary["completed"] == 1
    assert summary["pending"] == 1
    assert summary["correct"] == 0
    assert summary["accuracy"] == pytest.approx(0.0)
    assert summary["average_confidence"] == pytest.approx(70.0)
    assert summary["up_count"] == 1
    assert summary["down_count"] == 0
    assert summary["flat_count"] == 1


def test_format_timeline_empty_state_includes_filters() -> None:
    message = format_timeline_empty_state("1h", "4h", "7d")
    assert "1h / 4h / 7d" in message
    assert "No timeline events match the current filters" in message


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
    marker_traces = {
        trace.name: trace for trace in fig.data if getattr(trace, "mode", "") == "markers"
    }

    assert set(marker_traces.keys()) == {
        "UP - Realized (Correct)",
        "DOWN - Realized (Wrong)",
        "UP - Pending",
    }
    assert marker_traces["UP - Realized (Correct)"].marker.opacity == 1.0
    assert marker_traces["DOWN - Realized (Wrong)"].marker.opacity == 0.4
    assert marker_traces["UP - Pending"].marker.opacity == 0.75
    assert marker_traces["UP - Realized (Correct)"].marker.size == 14
    assert marker_traces["DOWN - Realized (Wrong)"].marker.size == 12
    assert marker_traces["UP - Pending"].marker.size == 10
    assert marker_traces["UP - Realized (Correct)"].customdata[0][3] == "Realized (Correct)"
    assert marker_traces["DOWN - Realized (Wrong)"].customdata[0][3] == "Realized (Wrong)"
    assert marker_traces["UP - Pending"].customdata[0][3] == "Pending"


@pytest.mark.skipif(not _has_plotly, reason="plotly not installed")
def test_build_timeline_figure_confidence_exact_percent_and_na() -> None:
    from bitbat.gui.timeline import build_timeline_figure

    predictions = pd.DataFrame({
        "timestamp_utc": pd.date_range("2024-01-01", periods=2, freq="h"),
        "predicted_direction": ["up", "down"],
        "p_up": [0.7234, None],
        "p_down": [0.1, None],
        "predicted_return": [0.01, -0.005],
        "actual_return": [None, None],
    })
    prices = pd.DataFrame(
        {"close": [100.0, 101.0]},
        index=pd.date_range("2024-01-01", periods=2, freq="h"),
    )

    fig = build_timeline_figure(predictions, prices)
    marker_traces = {
        trace.name: trace for trace in fig.data if getattr(trace, "mode", "") == "markers"
    }

    assert marker_traces["UP - Pending"].customdata[0][0] == "72.34%"
    assert marker_traces["DOWN - Pending"].customdata[0][0] == "n/a"


@pytest.mark.skipif(not _has_plotly, reason="plotly not installed")
def test_build_timeline_figure_confidence_does_not_affect_marker_size() -> None:
    from bitbat.gui.timeline import build_timeline_figure

    predictions = pd.DataFrame({
        "timestamp_utc": pd.date_range("2024-01-01", periods=2, freq="h"),
        "predicted_direction": ["up", "up"],
        "p_up": [0.95, 0.51],
        "p_down": [0.01, 0.45],
        "correct": [True, True],
    })
    prices = pd.DataFrame(
        {"close": [100.0, 100.5]},
        index=pd.date_range("2024-01-01", periods=2, freq="h"),
    )

    fig = build_timeline_figure(predictions, prices)
    marker_traces = [trace for trace in fig.data if getattr(trace, "mode", "") == "markers"]
    assert len(marker_traces) == 1
    assert marker_traces[0].marker.size == 14


def test_build_timeline_overlay_frame_pending_semantics() -> None:
    predictions = pd.DataFrame({
        "timestamp_utc": pd.date_range("2024-01-01", periods=3, freq="h"),
        "predicted_direction": ["up", "down", "up"],
        "predicted_return": [0.01, -0.006, 0.004],
        "actual_return": [0.008, -0.003, None],
        "correct": [1, 0, None],
    })

    overlay = build_timeline_overlay_frame(predictions)
    assert len(overlay) == 3
    assert overlay.loc[2, "prediction_status"] == "pending"
    assert pd.isna(overlay.loc[2, "actual_return"])
    assert pd.isna(overlay.loc[2, "mismatch_abs"])
    assert overlay.loc[0, "mismatch_abs"] == pytest.approx(0.002)


@pytest.mark.skipif(not _has_plotly, reason="plotly not installed")
def test_build_timeline_figure_default_excludes_overlay_traces() -> None:
    from bitbat.gui.timeline import build_timeline_figure

    predictions = pd.DataFrame({
        "timestamp_utc": pd.date_range("2024-01-01", periods=3, freq="h"),
        "predicted_direction": ["up", "down", "flat"],
        "predicted_return": [0.01, -0.005, 0.002],
        "actual_return": [0.008, -0.002, None],
        "correct": [1, 0, None],
    })
    prices = pd.DataFrame(
        {"close": [100.0, 99.0, 101.0]},
        index=pd.date_range("2024-01-01", periods=3, freq="h"),
    )

    fig = build_timeline_figure(predictions, prices, show_overlay=False)
    names = {trace.name for trace in fig.data if trace.name}

    assert "Predicted Return" not in names
    assert "Realized Return" not in names
    assert "Mismatch Band" not in names


@pytest.mark.skipif(not _has_plotly, reason="plotly not installed")
def test_build_timeline_figure_readability_dense_data_uses_grouped_marker_traces() -> None:
    from bitbat.gui.timeline import build_timeline_figure

    ts = pd.date_range("2024-01-01", periods=36, freq="h")
    predictions = pd.DataFrame({
        "timestamp_utc": ts,
        "predicted_direction": ["up", "down", "flat"] * 12,
        "correct": [True, False, None] * 12,
        "p_up": [0.8, 0.2, None] * 12,
        "p_down": [0.1, 0.7, None] * 12,
        "predicted_return": [0.004, -0.005, 0.0] * 12,
    })
    prices = pd.DataFrame(
        {"close": [40_000 + idx * 10 for idx in range(len(ts))]},
        index=ts,
    )

    fig = build_timeline_figure(predictions, prices)
    marker_traces = [trace for trace in fig.data if getattr(trace, "mode", "") == "markers"]
    total_points = sum(len(trace.x) for trace in marker_traces)

    assert total_points == len(predictions)
    assert len(marker_traces) <= 9
    assert len(marker_traces) < len(predictions)


@pytest.mark.skipif(not _has_plotly, reason="plotly not installed")
def test_build_timeline_comparison_figure_traces_and_pending_semantics() -> None:
    predictions = pd.DataFrame({
        "timestamp_utc": pd.date_range("2024-01-01", periods=4, freq="h"),
        "predicted_direction": ["up", "down", "flat", "up"],
        "predicted_return": [0.01, -0.005, 0.002, 0.004],
        "actual_return": [0.008, -0.002, None, None],
        "correct": [1, 0, None, None],
    })

    fig = build_timeline_comparison_figure(predictions)
    names = [trace.name for trace in fig.data if trace.name]
    assert "Predicted Return" in names
    assert "Realized Return" in names
    assert "Mismatch Band" in names

    realized_trace = next(trace for trace in fig.data if trace.name == "Realized Return")
    assert pd.isna(realized_trace.y[-1])
    assert pd.isna(realized_trace.y[-2])


@pytest.mark.skipif(not _has_plotly, reason="plotly not installed")
def test_build_timeline_figure_with_overlay_traces() -> None:
    from bitbat.gui.timeline import build_timeline_figure

    predictions = pd.DataFrame({
        "timestamp_utc": pd.date_range("2024-01-01", periods=3, freq="h"),
        "predicted_direction": ["up", "down", "flat"],
        "predicted_return": [0.01, -0.005, 0.002],
        "actual_return": [0.008, -0.002, None],
        "correct": [1, 0, None],
    })
    prices = pd.DataFrame(
        {"close": [100.0, 99.0, 101.0]},
        index=pd.date_range("2024-01-01", periods=3, freq="h"),
    )

    fig = build_timeline_figure(predictions, prices, show_overlay=True)
    names = [trace.name for trace in fig.data if trace.name]
    assert "Predicted Return" in names
    assert "Realized Return" in names
    assert "Mismatch Band" in names

    realized_trace = next(trace for trace in fig.data if trace.name == "Realized Return")
    assert pd.isna(realized_trace.y[-1])


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
    marker_traces = {
        trace.name: trace for trace in fig.data if getattr(trace, "mode", "") == "markers"
    }

    assert len(marker_traces) == 2
    assert marker_traces["UP - Pending"].y[0] == pytest.approx(39_900.0)
    assert marker_traces["DOWN - Pending"].y[0] == pytest.approx(40_250.0)
