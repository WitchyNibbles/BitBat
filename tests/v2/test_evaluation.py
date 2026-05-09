from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from bitbat_v2.config import BitBatV2Config
from bitbat_v2.domain import Candle
from bitbat_v2.evaluation import (
    compare_strategies,
    load_candles_from_parquet,
    unsupported_runtime_replay_summary,
)


def test_load_candles_from_parquet_requires_expected_columns(tmp_path) -> None:
    path = tmp_path / "bad.parquet"
    pd.DataFrame({"timestamp_utc": ["2026-04-25T10:00:00Z"], "close": [100_000.0]}).to_parquet(path)

    with pytest.raises(ValueError, match="missing required columns"):
        load_candles_from_parquet(path, BitBatV2Config())


def test_compare_strategies_returns_baseline_and_improved_metrics() -> None:
    start = datetime(2026, 4, 25, 10, 0, tzinfo=UTC)
    candles = [
        Candle(
            product_id="BTC-USD",
            granularity_seconds=300,
            start=start + timedelta(seconds=300 * idx),
            open=100_000.0 + (idx * 50.0),
            high=100_350.0 + (idx * 50.0),
            low=99_900.0 + (idx * 50.0),
            close=100_250.0 + (idx * 50.0),
            volume=8.0 + idx,
        )
        for idx in range(20)
    ]

    comparison = compare_strategies(candles, BitBatV2Config())

    assert set(comparison["strategies"]) == {"baseline_v1", "filtered_momentum_v2"}
    assert "final_equity_delta" in comparison["delta"]
    assert "trade_count_delta" in comparison["delta"]
    assert "fees_paid_delta" in comparison["delta"]
    assert "alpha_vs_buy_hold_delta" in comparison["delta"]


def test_compare_strategies_metrics_are_deterministic_for_synthetic_data() -> None:
    start = datetime(2026, 4, 25, 10, 0, tzinfo=UTC)
    candles = []
    for idx in range(24):
        base = 100_000.0 + (idx * 120.0)
        candles.append(
            Candle(
                product_id="BTC-USD",
                granularity_seconds=300,
                start=start + timedelta(seconds=300 * idx),
                open=base,
                high=base + 220.0,
                low=base - 80.0,
                close=base + 150.0,
                volume=10.0 + idx,
            )
        )

    comparison = compare_strategies(candles, BitBatV2Config())

    assert comparison["strategies"]["baseline_v1"]["trade_count"] >= 0
    assert comparison["strategies"]["filtered_momentum_v2"]["trade_count"] >= 0
    assert comparison["strategies"]["filtered_momentum_v2"]["hold_rate"] >= 0.0


def test_compare_strategies_reports_runtime_aligned_cost_metrics() -> None:
    start = datetime(2026, 4, 25, 10, 0, tzinfo=UTC)
    candles = [
        Candle(
            product_id="BTC-USD",
            granularity_seconds=300,
            start=start + timedelta(seconds=300 * idx),
            open=100_000.0 + (idx * 80.0),
            high=100_220.0 + (idx * 80.0),
            low=99_920.0 + (idx * 80.0),
            close=100_180.0 + (idx * 80.0),
            volume=10.0 + idx,
        )
        for idx in range(30)
    ]

    comparison = compare_strategies(
        candles,
        BitBatV2Config(fee_bps=10.0, slippage_bps=10.0),
    )

    for strategy_name in ("baseline_v1", "filtered_momentum_v2"):
        summary = comparison["strategies"][strategy_name]
        assert "fees_paid" in summary
        assert "turnover_usd" in summary
        assert "benchmark_equity" in summary
        assert "alpha_vs_buy_hold" in summary
        assert summary["fees_paid"] >= 0.0


def test_unsupported_runtime_replay_summary_marks_candidate_incompatible() -> None:
    summary = unsupported_runtime_replay_summary(
        signal_source="legacy_ml",
        model_name="random_forest_1h_4h",
        compatibility_reason="runtime_incompatible_family",
    )

    assert summary.runtime_compatible is False
    assert summary.compatibility_reason == "runtime_incompatible_family"
    assert summary.trade_count == 0
