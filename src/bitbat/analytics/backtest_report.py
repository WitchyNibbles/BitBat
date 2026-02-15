"""
Plain-language backtest reporting for BitBat.

Wraps the raw output of ``bitbat.backtest.engine.run`` and provides
human-readable summaries, ratings, and multi-scenario comparison.
No disk I/O — all metrics are computed in memory.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class BacktestReport:
    """Plain-language analysis of a completed BitBat backtest.

    Parameters
    ----------
    equity_curve:
        Equity series (cumulative) indexed by timestamp, starting at 1.0.
        Output of ``backtest.engine.run``.
    trades:
        Trades DataFrame with ``close``, ``position``, ``returns``, ``pnl``
        columns.  Output of ``backtest.engine.run``.
    preset_name:
        Human-readable name for the scenario (e.g. "Conservative").
    enter_threshold:
        Entry probability threshold used in this run.
    allow_short:
        Whether short positions were enabled.
    """

    def __init__(
        self,
        equity_curve: pd.Series,
        trades: pd.DataFrame,
        *,
        preset_name: str = "Custom",
        enter_threshold: float = 0.65,
        allow_short: bool = False,
    ) -> None:
        self.equity_curve = equity_curve
        self.trades = trades
        self.preset_name = preset_name
        self.enter_threshold = enter_threshold
        self.allow_short = allow_short
        self._metrics: dict[str, float] | None = None

    # ------------------------------------------------------------------
    # Core metrics (computed entirely in memory — no disk I/O)
    # ------------------------------------------------------------------

    def metrics(self) -> dict[str, float]:
        """Compute and cache backtest performance metrics.

        Returns
        -------
        dict with keys: sharpe, max_drawdown, hit_rate, avg_return,
        total_return, n_trades, turnover.
        """
        if self._metrics is not None:
            return self._metrics

        eq = self.equity_curve
        returns = eq.pct_change().fillna(0.0)

        # Sharpe (annualised, assuming hourly bars → 8760 periods/yr)
        std = float(returns.std())
        sharpe = float(np.sqrt(8760) * returns.mean() / std) if std != 0 else 0.0

        # Max drawdown
        drawdown = eq / eq.cummax() - 1
        max_dd = float(drawdown.min())

        # Hit rate (active bars only)
        nonzero = returns[returns != 0]
        hit_rate = float((nonzero > 0).sum() / len(nonzero)) if len(nonzero) else 0.0

        # Returns
        avg_ret = float(returns.mean())
        total_ret = float(eq.iloc[-1] / eq.iloc[0] - 1) if len(eq) > 0 else 0.0

        # Trade count and turnover
        pos = (
            self.trades["position"]
            if "position" in self.trades.columns
            else pd.Series(dtype=float)
        )
        changes = pos.diff().fillna(0.0)
        n_trades = int(changes.abs().gt(0).sum())
        turnover = float(changes.abs().sum())

        self._metrics = {
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "hit_rate": hit_rate,
            "avg_return": avg_ret,
            "total_return": total_ret,
            "n_trades": float(n_trades),
            "turnover": turnover,
        }
        return self._metrics

    # ------------------------------------------------------------------
    # Rating
    # ------------------------------------------------------------------

    def rating(self) -> str:
        """Return a one-word performance rating.

        Scale:

        - **Excellent**: Sharpe ≥ 1.5 and max_drawdown > −0.20
        - **Good**:      Sharpe ≥ 0.8 and max_drawdown > −0.30
        - **Fair**:      Sharpe ≥ 0.3
        - **Poor**:      otherwise
        """
        m = self.metrics()
        s = m["sharpe"]
        dd = m["max_drawdown"]
        if s >= 1.5 and dd > -0.20:
            return "Excellent"
        if s >= 0.8 and dd > -0.30:
            return "Good"
        if s >= 0.3:
            return "Fair"
        return "Poor"

    # ------------------------------------------------------------------
    # Plain-English summary
    # ------------------------------------------------------------------

    def plain_summary(self) -> str:
        """Return a Markdown plain-language performance summary."""
        m = self.metrics()
        r = self.rating()
        total_ret_pct = m["total_return"] * 100
        hit_pct = m["hit_rate"] * 100
        dd_pct = abs(m["max_drawdown"]) * 100

        rating_emoji = {
            "Excellent": "🌟",
            "Good": "✅",
            "Fair": "⚠️",
            "Poor": "❌",
        }.get(r, "")

        lines = [
            f"### {rating_emoji} {self.preset_name} — **{r}**",
            "",
            f"**Total Return**: {total_ret_pct:+.1f}%  ",
            f"**Sharpe Ratio**: {m['sharpe']:.2f}  ",
            f"**Max Drawdown**: −{dd_pct:.1f}%  ",
            f"**Win Rate**: {hit_pct:.1f}% of active periods  ",
            f"**Trades Made**: {int(m['n_trades']):,}  ",
            "",
        ]

        if r == "Excellent":
            lines.append(
                "The strategy performed **excellently** — strong risk-adjusted returns."
            )
        elif r == "Good":
            lines.append(
                "The strategy performed **well** — positive risk-adjusted returns."
            )
        elif r == "Fair":
            lines.append(
                "The strategy showed **modest** results — some edge but room to improve."
            )
        else:
            lines.append(
                "The strategy **underperformed** — consider adjusting the entry threshold."
            )

        if self.enter_threshold > 0.70 and int(m["n_trades"]) < 10:
            lines.append(
                "\n> **Tip:** The confidence threshold is very high — few trades were taken. "
                "Lowering it may improve coverage."
            )
        elif self.enter_threshold < 0.55 and m["max_drawdown"] < -0.25:
            lines.append(
                "\n> **Tip:** The low threshold led to frequent trading and larger drawdowns. "
                "Consider raising it for better risk control."
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # DataFrame row
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Return a one-row DataFrame for scenario comparison tables."""
        m = self.metrics()
        return pd.DataFrame(
            [
                {
                    "Scenario": self.preset_name,
                    "Threshold": f"{self.enter_threshold:.0%}",
                    "Allow Short": "Yes" if self.allow_short else "No",
                    "Total Return": f"{m['total_return'] * 100:+.1f}%",
                    "Sharpe": round(m["sharpe"], 2),
                    "Max Drawdown": f"-{abs(m['max_drawdown']) * 100:.1f}%",
                    "Win Rate": f"{m['hit_rate'] * 100:.1f}%",
                    "Trades": int(m["n_trades"]),
                    "Rating": self.rating(),
                }
            ]
        )


# ---------------------------------------------------------------------------
# Scenario comparison helper
# ---------------------------------------------------------------------------


def compare_scenarios(reports: list[BacktestReport]) -> pd.DataFrame:
    """Combine multiple BacktestReport objects into a comparison DataFrame.

    Parameters
    ----------
    reports:
        List of BacktestReport objects, one per scenario.

    Returns
    -------
    DataFrame with one row per scenario and the same columns as
    ``BacktestReport.to_dataframe()``.
    """
    if not reports:
        return pd.DataFrame()
    return pd.concat([r.to_dataframe() for r in reports], ignore_index=True)
