"""
Phase 3 Complete Integration Test.

Exercises the full analytics pipeline end-to-end:
  backtest_report → scenario comparison
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bitbat.analytics.backtest_report import BacktestReport, compare_scenarios
from bitbat.backtest.engine import run as backtest_run

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def dataset_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write a realistic synthetic feature dataset parquet."""
    rng = np.random.default_rng(2024)
    n = 300
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")

    data = {
        "timestamp_utc": idx,
        "feat_ret_1": rng.normal(0, 0.01, n),
        "feat_ret_4": rng.normal(0, 0.015, n),
        "feat_vol_24": rng.uniform(0.01, 0.05, n),
        "feat_rsi_14": rng.uniform(20, 80, n),
        "feat_sent_1h_mean": rng.normal(0, 0.2, n),
        "feat_sent_4h_mean": rng.normal(0, 0.2, n),
        "label": rng.choice(["up", "down", "flat"], size=n, p=[0.35, 0.35, 0.30]),
        "r_forward": rng.normal(0, 0.01, n),
    }
    df = pd.DataFrame(data)
    path = tmp_path_factory.mktemp("data") / "dataset.parquet"
    df.to_parquet(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Phase 3 end-to-end integration
# ---------------------------------------------------------------------------


class TestPhase3Integration:
    def test_backtest_engine_runs(self, dataset_path: Path) -> None:
        df = pd.read_parquet(dataset_path)
        n = len(df)
        rng = np.random.default_rng(42)
        idx = df.index[:n]
        close = pd.Series(np.cumprod(1 + df["feat_ret_1"].fillna(0)) * 100, index=idx)
        predicted_returns = pd.Series(rng.normal(0, 0.005, n), index=idx)
        trades, equity = backtest_run(close, predicted_returns)
        assert len(equity) == n
        assert "position" in trades.columns

    def test_backtest_report_from_engine(self, dataset_path: Path) -> None:
        df = pd.read_parquet(dataset_path)
        n = len(df)
        rng = np.random.default_rng(1)
        idx = df.index
        close = pd.Series(np.cumprod(1 + df["feat_ret_1"].fillna(0)) * 100, index=idx)
        predicted_returns = pd.Series(rng.normal(0, 0.005, n), index=idx)
        trades, equity = backtest_run(close, predicted_returns)
        report = BacktestReport(equity, trades, preset_name="Integration")
        m = report.metrics()
        assert "sharpe" in m
        assert "total_return" in m
        assert isinstance(report.plain_summary(), str)
        assert report.rating() in ("Excellent", "Good", "Fair", "Poor")

    def test_scenario_comparison_three_presets(self, dataset_path: Path) -> None:
        df = pd.read_parquet(dataset_path)
        n = len(df)
        rng = np.random.default_rng(42)
        idx = df.index
        close = pd.Series(np.cumprod(1 + df["feat_ret_1"].fillna(0)) * 100, index=idx)
        predicted_returns = pd.Series(rng.normal(0, 0.005, n), index=idx)

        signals = [0.005, 0.002, 0.001]
        names = ["Conservative", "Balanced", "Aggressive"]
        reports = []
        for sig, name in zip(signals, names, strict=False):
            t, eq = backtest_run(close, predicted_returns, min_signal=sig)
            reports.append(BacktestReport(eq, t, preset_name=name, enter_threshold=sig))

        comparison = compare_scenarios(reports)
        assert len(comparison) == 3
        assert list(comparison["Scenario"]) == names

    def test_full_pipeline_no_exceptions(self, dataset_path: Path) -> None:
        """End-to-end smoke test: backtest → comparison."""
        df = pd.read_parquet(dataset_path)

        # Backtest
        rng = np.random.default_rng(0)
        n = len(df)
        idx = df.index
        close = pd.Series(np.cumprod(1 + df["feat_ret_1"].fillna(0)) * 100, index=idx)
        predicted_returns = pd.Series(rng.normal(0, 0.005, n), index=idx)
        trades, equity = backtest_run(close, predicted_returns)

        # Report
        report = BacktestReport(equity, trades, preset_name="Smoke Test")
        assert report.rating() in ("Excellent", "Good", "Fair", "Poor")
        summary = report.plain_summary()
        assert "Smoke Test" in summary
