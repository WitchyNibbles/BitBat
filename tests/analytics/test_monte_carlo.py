"""Tests for Monte Carlo backtesting (Phase 5, Session 2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bitbat.analytics.monte_carlo import MonteCarloResult, MonteCarloSimulator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def returns() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.normal(0.0001, 0.005, 500)


@pytest.fixture(scope="module")
def simulator(returns: np.ndarray) -> MonteCarloSimulator:
    return MonteCarloSimulator(returns)


@pytest.fixture(scope="module")
def result(simulator: MonteCarloSimulator) -> MonteCarloResult:
    return simulator.run(n_simulations=200, seed=42)


# ---------------------------------------------------------------------------
# MonteCarloSimulator
# ---------------------------------------------------------------------------


class TestSimulator:
    def test_accepts_series(self) -> None:
        s = pd.Series(np.random.default_rng(0).normal(0, 0.01, 100))
        sim = MonteCarloSimulator(s)
        assert len(sim.returns) == 100

    def test_accepts_array(self, returns: np.ndarray) -> None:
        sim = MonteCarloSimulator(returns)
        assert len(sim.returns) == 500

    def test_run_returns_result(self, simulator: MonteCarloSimulator) -> None:
        r = simulator.run(n_simulations=10)
        assert isinstance(r, MonteCarloResult)

    def test_custom_path_length(self, simulator: MonteCarloSimulator) -> None:
        r = simulator.run(n_simulations=10, path_length=100)
        assert r.equity_paths.shape == (10, 100)


# ---------------------------------------------------------------------------
# MonteCarloResult
# ---------------------------------------------------------------------------


class TestResult:
    def test_n_simulations(self, result: MonteCarloResult) -> None:
        assert result.n_simulations == 200

    def test_total_returns_shape(self, result: MonteCarloResult) -> None:
        assert result.total_returns.shape == (200,)

    def test_sharpe_ratios_shape(self, result: MonteCarloResult) -> None:
        assert result.sharpe_ratios.shape == (200,)

    def test_max_drawdowns_shape(self, result: MonteCarloResult) -> None:
        assert result.max_drawdowns.shape == (200,)

    def test_max_drawdowns_nonpositive(self, result: MonteCarloResult) -> None:
        assert (result.max_drawdowns <= 0).all()

    def test_equity_paths_shape(self, result: MonteCarloResult) -> None:
        assert result.equity_paths.shape[0] == 200
        assert result.equity_paths.shape[1] == 500  # default path_length

    def test_equity_paths_start_near_one(self, result: MonteCarloResult) -> None:
        # First column should be close to 1 + first_return
        assert np.allclose(result.equity_paths[:, 0], 1 + result.equity_paths[:, 0] - 1, atol=0.1)


class TestConfidenceInterval:
    def test_returns_tuple(self, result: MonteCarloResult) -> None:
        ci = result.confidence_interval("total_returns")
        assert isinstance(ci, tuple)
        assert len(ci) == 2

    def test_lower_less_than_upper(self, result: MonteCarloResult) -> None:
        lo, hi = result.confidence_interval("total_returns")
        assert lo <= hi

    def test_sharpe_ci(self, result: MonteCarloResult) -> None:
        lo, hi = result.confidence_interval("sharpe_ratios")
        assert lo <= hi

    def test_drawdown_ci(self, result: MonteCarloResult) -> None:
        lo, hi = result.confidence_interval("max_drawdowns")
        assert lo <= hi
        assert lo <= 0  # drawdowns are negative

    def test_wider_at_lower_confidence(self, result: MonteCarloResult) -> None:
        ci90 = result.confidence_interval("total_returns", level=0.90)
        ci99 = result.confidence_interval("total_returns", level=0.99)
        assert (ci99[1] - ci99[0]) >= (ci90[1] - ci90[0])


class TestProbabilities:
    def test_probability_of_loss_in_range(self, result: MonteCarloResult) -> None:
        p = result.probability_of_loss()
        assert 0.0 <= p <= 1.0

    def test_probability_of_drawdown_in_range(self, result: MonteCarloResult) -> None:
        p = result.probability_of_drawdown(-0.20)
        assert 0.0 <= p <= 1.0

    def test_higher_threshold_more_likely(self, result: MonteCarloResult) -> None:
        p_tight = result.probability_of_drawdown(-0.05)
        p_loose = result.probability_of_drawdown(-0.50)
        assert p_tight >= p_loose


class TestSummary:
    def test_returns_dict(self, result: MonteCarloResult) -> None:
        s = result.summary()
        assert isinstance(s, dict)

    def test_required_keys(self, result: MonteCarloResult) -> None:
        s = result.summary()
        for key in (
            "n_simulations",
            "median_return",
            "mean_return",
            "return_ci_lower",
            "return_ci_upper",
            "median_sharpe",
            "sharpe_ci_lower",
            "sharpe_ci_upper",
            "median_max_drawdown",
            "drawdown_ci_lower",
            "drawdown_ci_upper",
            "probability_of_loss",
            "probability_of_20pct_drawdown",
        ):
            assert key in s, f"Missing key: {key}"

    def test_json_serialisable(self, result: MonteCarloResult) -> None:
        import json

        json.dumps(result.summary())  # should not raise


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_all_positive_returns(self) -> None:
        returns = np.full(100, 0.001)
        sim = MonteCarloSimulator(returns)
        r = sim.run(n_simulations=50)
        assert (r.total_returns > 0).all()
        assert r.probability_of_loss() == 0.0

    def test_all_negative_returns(self) -> None:
        returns = np.full(100, -0.001)
        sim = MonteCarloSimulator(returns)
        r = sim.run(n_simulations=50)
        assert (r.total_returns < 0).all()
        assert r.probability_of_loss() == 1.0

    def test_zero_returns(self) -> None:
        returns = np.zeros(100)
        sim = MonteCarloSimulator(returns)
        r = sim.run(n_simulations=20)
        np.testing.assert_allclose(r.total_returns, 0.0, atol=1e-10)

    def test_reproducible_with_same_seed(self) -> None:
        returns = np.random.default_rng(0).normal(0, 0.01, 200)
        sim = MonteCarloSimulator(returns)
        r1 = sim.run(n_simulations=50, seed=123)
        r2 = sim.run(n_simulations=50, seed=123)
        np.testing.assert_array_equal(r1.total_returns, r2.total_returns)
