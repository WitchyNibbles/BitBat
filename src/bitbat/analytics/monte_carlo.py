"""
Monte Carlo simulation for backtest robustness analysis.

Bootstrap-resamples trade returns to estimate the distribution of
outcomes (total return, Sharpe ratio, max drawdown) and compute
confidence intervals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MonteCarloResult:
    """Container for Monte Carlo simulation output.

    Attributes
    ----------
    total_returns : np.ndarray
        Simulated total returns, one per path.
    sharpe_ratios : np.ndarray
        Annualised Sharpe ratios per path.
    max_drawdowns : np.ndarray
        Maximum drawdown per path (negative values).
    equity_paths : np.ndarray
        2-D array of shape ``(n_simulations, n_steps)`` with cumulative equity.
    n_simulations : int
        Number of paths simulated.
    """

    total_returns: np.ndarray
    sharpe_ratios: np.ndarray
    max_drawdowns: np.ndarray
    equity_paths: np.ndarray
    n_simulations: int

    def confidence_interval(
        self,
        metric: str = "total_returns",
        level: float = 0.95,
    ) -> tuple[float, float]:
        """Return the symmetric confidence interval for *metric*.

        Parameters
        ----------
        metric : str
            One of ``"total_returns"``, ``"sharpe_ratios"``, ``"max_drawdowns"``.
        level : float
            Confidence level (default 0.95 → 95%).

        Returns
        -------
        (lower, upper) tuple.
        """
        arr = getattr(self, metric)
        alpha = (1 - level) / 2
        return float(np.percentile(arr, alpha * 100)), float(np.percentile(arr, (1 - alpha) * 100))

    def probability_of_loss(self) -> float:
        """Fraction of simulations that ended with negative total return."""
        return float((self.total_returns < 0).mean())

    def probability_of_drawdown(self, threshold: float = -0.20) -> float:
        """Fraction of simulations where max drawdown exceeded *threshold*."""
        return float((self.max_drawdowns < threshold).mean())

    def summary(self) -> dict[str, float]:
        """Return a JSON-serialisable summary dict."""
        ci_ret = self.confidence_interval("total_returns")
        ci_sharpe = self.confidence_interval("sharpe_ratios")
        ci_dd = self.confidence_interval("max_drawdowns")
        return {
            "n_simulations": self.n_simulations,
            "median_return": float(np.median(self.total_returns)),
            "mean_return": float(np.mean(self.total_returns)),
            "return_ci_lower": ci_ret[0],
            "return_ci_upper": ci_ret[1],
            "median_sharpe": float(np.median(self.sharpe_ratios)),
            "sharpe_ci_lower": ci_sharpe[0],
            "sharpe_ci_upper": ci_sharpe[1],
            "median_max_drawdown": float(np.median(self.max_drawdowns)),
            "drawdown_ci_lower": ci_dd[0],
            "drawdown_ci_upper": ci_dd[1],
            "probability_of_loss": self.probability_of_loss(),
            "probability_of_20pct_drawdown": self.probability_of_drawdown(-0.20),
        }


class MonteCarloSimulator:
    """Bootstrap Monte Carlo simulation over backtest returns.

    Parameters
    ----------
    returns : pd.Series | np.ndarray
        Per-bar PnL returns from a backtest (e.g. ``trades["pnl"]``).
    periods_per_year : int
        Number of bars per year for Sharpe annualisation (default 8760 for hourly).
    """

    def __init__(
        self,
        returns: pd.Series | np.ndarray,
        *,
        periods_per_year: int = 8760,
    ) -> None:
        self.returns = np.asarray(returns, dtype=float)
        self.periods_per_year = periods_per_year

    def run(
        self,
        n_simulations: int = 1000,
        path_length: int | None = None,
        seed: int = 42,
    ) -> MonteCarloResult:
        """Run the bootstrap simulation.

        Parameters
        ----------
        n_simulations : int
            Number of random equity paths to generate.
        path_length : int | None
            Steps per path; defaults to ``len(self.returns)``.
        seed : int
            RNG seed for reproducibility.

        Returns
        -------
        MonteCarloResult with arrays of simulated metrics.
        """
        rng = np.random.default_rng(seed)
        n = path_length or len(self.returns)

        # Bootstrap: sample returns with replacement
        sampled = rng.choice(self.returns, size=(n_simulations, n), replace=True)

        # Cumulative equity paths
        equity_paths = np.cumprod(1 + sampled, axis=1)

        # Total return per path
        total_returns = equity_paths[:, -1] - 1.0

        # Sharpe ratio per path
        means = sampled.mean(axis=1)
        stds = sampled.std(axis=1)
        stds[stds == 0] = np.inf
        sharpe_ratios = np.sqrt(self.periods_per_year) * means / stds

        # Max drawdown per path
        cummax = np.maximum.accumulate(equity_paths, axis=1)
        drawdowns = equity_paths / cummax - 1.0
        max_drawdowns = drawdowns.min(axis=1)

        return MonteCarloResult(
            total_returns=total_returns,
            sharpe_ratios=sharpe_ratios,
            max_drawdowns=max_drawdowns,
            equity_paths=equity_paths,
            n_simulations=n_simulations,
        )
