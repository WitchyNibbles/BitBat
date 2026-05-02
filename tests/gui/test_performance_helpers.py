from __future__ import annotations

import pandas as pd
import pytest

from bitbat.gui.performance import (
    attach_price_evidence,
    build_accuracy_history,
    format_recent_predictions,
    normalize_performance_rows,
    resolve_performance_scope,
    summarize_current_streak,
    summarize_performance_rows,
    summarize_recent_mix,
)


def test_resolve_performance_scope_prefers_active_session_pair() -> None:
    freq, horizon = resolve_performance_scope(
        {"active_freq": "5m", "active_horizon": "30m"},
        {"freq": "1h", "horizon": "4h"},
    )

    assert (freq, horizon) == ("5m", "30m")


def test_normalize_performance_rows_uses_p_flat_for_confidence() -> None:
    df = pd.DataFrame([
        {
            "timestamp_utc": "2026-03-31T12:10:00Z",
            "predicted_direction": "flat",
            "actual_direction": "flat",
            "actual_return": 0.0,
            "correct": 1,
            "p_up": 0.0079,
            "p_down": 0.0041,
            "p_flat": 0.9879,
        }
    ])

    normalized = normalize_performance_rows(df)

    assert normalized.loc[0, "confidence"] == pytest.approx(0.9879)


def test_normalize_performance_rows_deduplicates_same_prediction_bar() -> None:
    df = pd.DataFrame([
        {
            "timestamp_utc": "2026-03-31T12:10:00Z",
            "created_at": "2026-05-02T12:20:00Z",
            "predicted_direction": "flat",
            "actual_direction": "flat",
            "actual_return": 0.0,
            "correct": 1,
            "p_up": 0.01,
            "p_down": 0.02,
            "p_flat": 0.97,
        },
        {
            "timestamp_utc": "2026-03-31T12:10:00Z",
            "created_at": "2026-05-02T12:25:00Z",
            "predicted_direction": "flat",
            "actual_direction": "flat",
            "actual_return": 0.0,
            "correct": 1,
            "p_up": 0.004,
            "p_down": 0.008,
            "p_flat": 0.988,
        },
        {
            "timestamp_utc": "2026-03-31T12:15:00Z",
            "created_at": "2026-05-02T12:30:00Z",
            "predicted_direction": "up",
            "actual_direction": "up",
            "actual_return": 0.01,
            "correct": 1,
            "p_up": 0.8,
            "p_down": 0.1,
            "p_flat": 0.1,
        },
    ])

    normalized = normalize_performance_rows(df)

    assert len(normalized) == 2
    assert normalized.loc[0, "confidence"] == pytest.approx(0.988)
    assert normalized.loc[1, "timestamp_utc"] == pd.Timestamp("2026-03-31T12:15:00")


def test_summarize_current_streak_uses_prediction_time_not_write_order() -> None:
    df = pd.DataFrame([
        {
            "timestamp_utc": "2026-03-31T12:00:00Z",
            "predicted_direction": "up",
            "actual_direction": "down",
            "actual_return": -0.01,
            "correct": 0,
        },
        {
            "timestamp_utc": "2026-03-31T12:05:00Z",
            "predicted_direction": "flat",
            "actual_direction": "flat",
            "actual_return": 0.0,
            "correct": 1,
        },
        {
            "timestamp_utc": "2026-03-31T12:10:00Z",
            "predicted_direction": "flat",
            "actual_direction": "flat",
            "actual_return": 0.0,
            "correct": 1,
        },
    ])

    streak = summarize_current_streak(df, limit=20)

    assert streak["type"] == "win"
    assert streak["count"] == 2


def test_summarize_performance_rows_counts_only_scoped_rows() -> None:
    df = pd.DataFrame([
        {
            "timestamp_utc": "2026-03-31T12:00:00Z",
            "predicted_direction": "up",
            "actual_direction": "up",
            "actual_return": 0.02,
            "correct": 1,
        },
        {
            "timestamp_utc": "2026-03-31T13:00:00Z",
            "predicted_direction": "down",
            "actual_direction": "up",
            "actual_return": 0.01,
            "correct": 0,
        },
        {
            "timestamp_utc": "2026-03-31T14:00:00Z",
            "predicted_direction": "flat",
            "actual_direction": pd.NA,
            "actual_return": pd.NA,
            "correct": pd.NA,
        },
    ])

    summary = summarize_performance_rows(df)

    assert summary == {
        "total": 3,
        "completed": 2,
        "correct": 1,
        "pending": 1,
        "accuracy": 50.0,
    }


def test_build_accuracy_history_returns_market_time_series() -> None:
    df = pd.DataFrame([
        {
            "timestamp_utc": "2026-03-31T12:00:00Z",
            "predicted_direction": "up",
            "actual_direction": "up",
            "actual_return": 0.01,
            "correct": 1,
        },
        {
            "timestamp_utc": "2026-03-31T12:05:00Z",
            "predicted_direction": "down",
            "actual_direction": "up",
            "actual_return": 0.01,
            "correct": 0,
        },
    ])

    history = build_accuracy_history(df, window=2)

    assert list(history["Accuracy %"]) == [100.0, 50.0]


def test_recent_mix_surfaces_flat_heavy_window() -> None:
    df = pd.DataFrame([
        {
            "timestamp_utc": "2026-03-31T12:00:00Z",
            "predicted_direction": "flat",
            "actual_direction": "flat",
            "actual_return": 0.0,
            "correct": 1,
        },
        {
            "timestamp_utc": "2026-03-31T12:05:00Z",
            "predicted_direction": "flat",
            "actual_direction": "flat",
            "actual_return": 0.0,
            "correct": 1,
        },
        {
            "timestamp_utc": "2026-03-31T12:10:00Z",
            "predicted_direction": "up",
            "actual_direction": "flat",
            "actual_return": 0.0,
            "correct": 0,
        },
    ])

    mix = summarize_recent_mix(df, limit=20)

    assert mix["window"] == 3
    assert mix["predicted_flat_rate"] == pytest.approx(66.6666666667)
    assert mix["actual_flat_rate"] == pytest.approx(100.0)


def test_format_recent_predictions_renders_flat_confidence_from_p_flat() -> None:
    df = pd.DataFrame([
        {
            "timestamp_utc": "2026-03-31T12:10:00Z",
            "predicted_direction": "flat",
            "actual_direction": "flat",
            "actual_return": 0.0,
            "correct": 1,
            "p_up": 0.004,
            "p_down": 0.008,
            "p_flat": 0.988,
        }
    ])

    formatted = format_recent_predictions(df, limit=20)

    assert formatted.loc[0, "Prediction"] == "➡️ FLAT"
    assert formatted.loc[0, "Confidence %"] == "98.8%"
    assert formatted.loc[0, "Predicted Price"] == "—"
    assert formatted.loc[0, "Actual Price"] == "—"


def test_format_recent_predictions_hides_duplicate_bars() -> None:
    df = pd.DataFrame([
        {
            "timestamp_utc": "2026-03-31T12:10:00Z",
            "created_at": "2026-05-02T12:20:00Z",
            "predicted_direction": "flat",
            "actual_direction": "flat",
            "actual_return": 0.0,
            "correct": 1,
            "p_up": 0.01,
            "p_down": 0.02,
            "p_flat": 0.97,
        },
        {
            "timestamp_utc": "2026-03-31T12:10:00Z",
            "created_at": "2026-05-02T12:25:00Z",
            "predicted_direction": "flat",
            "actual_direction": "flat",
            "actual_return": 0.0,
            "correct": 1,
            "p_up": 0.004,
            "p_down": 0.008,
            "p_flat": 0.988,
        },
    ])

    formatted = format_recent_predictions(df, limit=20)

    assert len(formatted) == 1
    assert formatted.loc[0, "Confidence %"] == "98.8%"


def test_attach_price_evidence_derives_actual_price_from_market_data() -> None:
    predictions = pd.DataFrame([
        {
            "timestamp_utc": "2026-03-31T12:10:00Z",
            "predicted_direction": "up",
            "actual_direction": "up",
            "actual_return": 0.02,
            "predicted_price": 102.0,
            "correct": 1,
        }
    ])
    prices = pd.DataFrame([
        {"timestamp_utc": "2026-03-31T12:10:00Z", "close": 100.0},
    ])

    enriched = attach_price_evidence(predictions, prices, freq="1h")

    assert enriched.loc[0, "entry_price"] == pytest.approx(100.0)
    assert enriched.loc[0, "actual_price"] == pytest.approx(102.0)
    assert enriched.loc[0, "price_gap_pct"] == pytest.approx(0.0)


def test_format_recent_predictions_shows_price_proof_columns() -> None:
    predictions = pd.DataFrame([
        {
            "timestamp_utc": "2026-03-31T12:10:00Z",
            "predicted_direction": "up",
            "actual_direction": "up",
            "actual_return": 0.02,
            "predicted_price": 103.0,
            "correct": 1,
            "p_up": 0.8,
            "p_down": 0.1,
            "p_flat": 0.1,
        }
    ])
    prices = pd.DataFrame([
        {"timestamp_utc": "2026-03-31T12:10:00Z", "close": 100.0},
    ])

    formatted = format_recent_predictions(predictions, limit=20, prices=prices, freq="1h")

    assert formatted.loc[0, "Predicted Price"] == "$103.00"
    assert formatted.loc[0, "Actual Price"] == "$102.00"
    assert formatted.loc[0, "Price Gap %"] == "1.0%"
    assert formatted.loc[0, "Actual Return"] == "2.0%"
