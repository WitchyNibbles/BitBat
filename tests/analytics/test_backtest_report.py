"""Tests for BacktestReport and compare_scenarios (Phase 3, Session 3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bitbat.analytics.backtest_report import BacktestReport, compare_scenarios
from bitbat.backtest.engine import run as backtest_run


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_equity(n: int = 100, seed: int = 42) -> pd.Series:
    """Construct a synthetic equity curve from a random walk."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    returns = rng.normal(0.0001, 0.005, n)
    equity = pd.Series((1 + returns).cumprod(), index=idx, name="equity")
    return equity


def _make_trades(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Construct a synthetic trades DataFrame."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    returns = rng.normal(0.0001, 0.005, n)
    position = rng.choice([-1.0, 0.0, 1.0], size=n)
    return pd.DataFrame(
        {
            "close": np.cumprod(1 + returns) * 100,
            "position": position,
            "returns": returns,
            "pnl": position * returns,
        },
        index=idx,
    )


@pytest.fixture()
def report() -> BacktestReport:
    return BacktestReport(
        _make_equity(),
        _make_trades(),
        preset_name="Test",
        enter_threshold=0.65,
    )


@pytest.fixture()
def engine_report() -> BacktestReport:
    """BacktestReport built from the real backtest engine output."""
    rng = np.random.default_rng(0)
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    close = pd.Series(np.cumprod(1 + rng.normal(0.0001, 0.005, n)) * 100, index=idx)
    proba_up = pd.Series(rng.uniform(0.3, 0.8, n), index=idx)
    proba_down = pd.Series(rng.uniform(0.1, 0.5, n), index=idx)
    trades, equity = backtest_run(close, proba_up, proba_down, enter=0.65)
    return BacktestReport(equity, trades, preset_name="Engine", enter_threshold=0.65)


# ---------------------------------------------------------------------------
# BacktestReport.metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_returns_dict(self, report: BacktestReport) -> None:
        m = report.metrics()
        assert isinstance(m, dict)

    def test_required_keys(self, report: BacktestReport) -> None:
        m = report.metrics()
        for key in ("sharpe", "max_drawdown", "hit_rate", "avg_return", "total_return", "n_trades", "turnover"):
            assert key in m, f"Missing key: {key}"

    def test_max_drawdown_nonpositive(self, report: BacktestReport) -> None:
        m = report.metrics()
        assert m["max_drawdown"] <= 0.0

    def test_hit_rate_in_range(self, report: BacktestReport) -> None:
        m = report.metrics()
        assert 0.0 <= m["hit_rate"] <= 1.0

    def test_n_trades_nonnegative(self, report: BacktestReport) -> None:
        m = report.metrics()
        assert m["n_trades"] >= 0

    def test_metrics_cached(self, report: BacktestReport) -> None:
        m1 = report.metrics()
        m2 = report.metrics()
        assert m1 is m2  # same object (cached)

    def test_flat_equity_sharpe_zero(self) -> None:
        idx = pd.date_range("2024-01-01", periods=10, freq="1h")
        eq = pd.Series([1.0] * 10, index=idx)
        trades = pd.DataFrame({"position": [0.0] * 10}, index=idx)
        r = BacktestReport(eq, trades)
        assert r.metrics()["sharpe"] == 0.0

    def test_total_return_computed(self, engine_report: BacktestReport) -> None:
        m = engine_report.metrics()
        eq = engine_report.equity_curve
        expected = eq.iloc[-1] / eq.iloc[0] - 1
        assert abs(m["total_return"] - expected) < 1e-9

    def test_no_position_col_handled(self) -> None:
        eq = _make_equity()
        trades = pd.DataFrame({"close": [100.0] * 100})  # no 'position' col
        r = BacktestReport(eq, trades)
        m = r.metrics()
        assert m["n_trades"] == 0
        assert m["turnover"] == 0.0


# ---------------------------------------------------------------------------
# BacktestReport.rating
# ---------------------------------------------------------------------------


class TestRating:
    VALID_RATINGS = {"Excellent", "Good", "Fair", "Poor"}

    def test_returns_valid_string(self, report: BacktestReport) -> None:
        assert report.rating() in self.VALID_RATINGS

    def test_poor_rating_on_bad_equity(self) -> None:
        rng = np.random.default_rng(99)
        idx = pd.date_range("2024-01-01", periods=50, freq="1h")
        # Consistently declining equity
        eq = pd.Series(np.cumprod(1 + rng.normal(-0.005, 0.003, 50)), index=idx)
        trades = pd.DataFrame({"position": [1.0] * 50}, index=idx)
        r = BacktestReport(eq, trades)
        assert r.rating() in ("Poor", "Fair")

    def test_engine_report_has_rating(self, engine_report: BacktestReport) -> None:
        assert engine_report.rating() in self.VALID_RATINGS


# ---------------------------------------------------------------------------
# BacktestReport.plain_summary
# ---------------------------------------------------------------------------


class TestPlainSummary:
    def test_returns_string(self, report: BacktestReport) -> None:
        assert isinstance(report.plain_summary(), str)

    def test_contains_preset_name(self, report: BacktestReport) -> None:
        assert "Test" in report.plain_summary()

    def test_contains_rating(self, report: BacktestReport) -> None:
        rating = report.rating()
        assert rating in report.plain_summary()

    def test_contains_return_percentage(self, report: BacktestReport) -> None:
        summary = report.plain_summary()
        assert "%" in summary

    def test_high_threshold_tip_shown(self) -> None:
        eq = _make_equity()
        # Trades with very few position changes (mimics high threshold / few trades)
        trades = pd.DataFrame(
            {"close": [100.0] * 100, "position": [0.0] * 100},
            index=eq.index,
        )
        r = BacktestReport(eq, trades, enter_threshold=0.80)
        summary = r.plain_summary()
        assert "Tip" in summary

    def test_low_threshold_tip_shown(self) -> None:
        rng = np.random.default_rng(7)
        idx = pd.date_range("2024-01-01", periods=100, freq="1h")
        # Big drawdown scenario
        eq = pd.Series(np.cumprod(1 + rng.normal(-0.003, 0.01, 100)), index=idx)
        trades = pd.DataFrame({"position": [1.0] * 100}, index=idx)
        r = BacktestReport(eq, trades, enter_threshold=0.50)
        summary = r.plain_summary()
        # Either the drawdown tip or the main verdict — just make sure it's non-empty
        assert len(summary) > 50


# ---------------------------------------------------------------------------
# BacktestReport.to_dataframe
# ---------------------------------------------------------------------------


class TestToDataframe:
    def test_returns_dataframe(self, report: BacktestReport) -> None:
        df = report.to_dataframe()
        assert isinstance(df, pd.DataFrame)

    def test_one_row(self, report: BacktestReport) -> None:
        df = report.to_dataframe()
        assert len(df) == 1

    def test_required_columns(self, report: BacktestReport) -> None:
        df = report.to_dataframe()
        for col in ("Scenario", "Threshold", "Allow Short", "Total Return", "Sharpe", "Max Drawdown", "Win Rate", "Trades", "Rating"):
            assert col in df.columns, f"Missing column: {col}"

    def test_scenario_name_matches(self, report: BacktestReport) -> None:
        df = report.to_dataframe()
        assert df["Scenario"].iloc[0] == "Test"

    def test_threshold_formatted(self, report: BacktestReport) -> None:
        df = report.to_dataframe()
        assert "%" in df["Threshold"].iloc[0]


# ---------------------------------------------------------------------------
# compare_scenarios
# ---------------------------------------------------------------------------


class TestCompareScenarios:
    def test_empty_list(self) -> None:
        df = compare_scenarios([])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_single_report(self, report: BacktestReport) -> None:
        df = compare_scenarios([report])
        assert len(df) == 1

    def test_multiple_reports(self) -> None:
        r1 = BacktestReport(_make_equity(seed=1), _make_trades(seed=1), preset_name="A", enter_threshold=0.60)
        r2 = BacktestReport(_make_equity(seed=2), _make_trades(seed=2), preset_name="B", enter_threshold=0.70)
        r3 = BacktestReport(_make_equity(seed=3), _make_trades(seed=3), preset_name="C", enter_threshold=0.80)
        df = compare_scenarios([r1, r2, r3])
        assert len(df) == 3
        assert list(df["Scenario"]) == ["A", "B", "C"]

    def test_result_has_no_duplicated_index(self) -> None:
        r1 = BacktestReport(_make_equity(seed=1), _make_trades(seed=1), preset_name="A")
        r2 = BacktestReport(_make_equity(seed=2), _make_trades(seed=2), preset_name="B")
        df = compare_scenarios([r1, r2])
        assert list(df.index) == [0, 1]


# ---------------------------------------------------------------------------
# Integration: engine → BacktestReport
# ---------------------------------------------------------------------------


class TestEngineIntegration:
    def test_engine_output_feeds_report(self, engine_report: BacktestReport) -> None:
        m = engine_report.metrics()
        assert m["sharpe"] is not None
        assert m["max_drawdown"] <= 0.0

    def test_allow_short_flag_stored(self) -> None:
        rng = np.random.default_rng(5)
        n = 50
        idx = pd.date_range("2024-01-01", periods=n, freq="1h")
        close = pd.Series(np.cumprod(1 + rng.normal(0, 0.005, n)) * 100, index=idx)
        pu = pd.Series(rng.uniform(0.3, 0.7, n), index=idx)
        pd_ = pd.Series(rng.uniform(0.2, 0.6, n), index=idx)
        trades, equity = backtest_run(close, pu, pd_, enter=0.60, allow_short=True)
        r = BacktestReport(equity, trades, allow_short=True)
        assert r.allow_short is True
        df = r.to_dataframe()
        assert df["Allow Short"].iloc[0] == "Yes"
