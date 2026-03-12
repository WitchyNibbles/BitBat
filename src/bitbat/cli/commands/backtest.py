"""Backtest CLI commands."""

from __future__ import annotations

import click
import pandas as pd

from bitbat.backtest.engine import run as run_strategy  # noqa: F401
from bitbat.backtest.metrics import summary as summarize_backtest  # noqa: F401
from bitbat.cli._helpers import (
    _config,
    _ensure_path_exists,
    _load_prices_indexed,
    _predictions_path,
    _resolve_setting,
)
from bitbat.contracts import ensure_predictions_contract


@click.group(help="Backtest utilities.")
def backtest() -> None:
    """Backtest command namespace."""


@backtest.command("run")
@click.option("--freq", default=None, help="Bar frequency.")
@click.option("--horizon", default=None, help="Prediction horizon.")
@click.option(
    "--allow-short",
    "allow_short_flag",
    is_flag=True,
    default=False,
    help="Enable short positions.",
)
@click.option(
    "--no-allow-short",
    "no_allow_short_flag",
    is_flag=True,
    default=False,
    help="Disable short positions.",
)
@click.option(
    "--cost-bps",
    "--cost_bps",
    type=float,
    default=None,
    help="Round-trip cost in basis points.",
)
@click.option(
    "--fee-bps",
    "--fee_bps",
    type=float,
    default=None,
    help="Transaction fee component in basis points.",
)
@click.option(
    "--slippage-bps",
    "--slippage_bps",
    type=float,
    default=None,
    help="Slippage component in basis points.",
)
def backtest_run(
    freq: str | None,
    horizon: str | None,
    allow_short_flag: bool,
    no_allow_short_flag: bool,
    cost_bps: float | None,
    fee_bps: float | None,
    slippage_bps: float | None,
) -> None:
    """Run backtest using stored predictions."""
    freq_val = _resolve_setting(freq, "freq")
    horizon_val = _resolve_setting(horizon, "horizon")
    cfg = _config()
    cost = cost_bps if cost_bps is not None else float(cfg["cost_bps"])
    resolved_fee_bps = fee_bps if fee_bps is not None else float(cfg.get("fee_bps", cost))
    resolved_slippage_bps = (
        slippage_bps if slippage_bps is not None else float(cfg.get("slippage_bps", 0.0))
    )

    if allow_short_flag and no_allow_short_flag:
        raise click.BadParameter("Specify only one of --allow-short or --no-allow-short.")

    predictions_path = _predictions_path(freq_val, horizon_val)
    _ensure_path_exists(predictions_path, "Predictions parquet")
    preds = ensure_predictions_contract(pd.read_parquet(predictions_path))
    if preds.empty:
        raise click.ClickException("No predictions available for backtest.")

    preds["timestamp_utc"] = pd.to_datetime(
        preds["timestamp_utc"], utc=True, errors="coerce"
    ).dt.tz_localize(None)
    preds = preds.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")
    prices = _load_prices_indexed(freq_val)
    close = prices["close"].reindex(preds["timestamp_utc"]).ffill()

    predicted_returns = preds.set_index("timestamp_utc")["predicted_return"]

    if allow_short_flag:
        allow_short_val = True
    elif no_allow_short_flag:
        allow_short_val = False
    else:
        allow_short_val = bool(_config().get("allow_short", False))

    trades, equity = run_strategy(
        close,
        predicted_returns,
        allow_short=allow_short_val,
        cost_bps=cost,
        fee_bps=resolved_fee_bps,
        slippage_bps=resolved_slippage_bps,
    )
    metrics = summarize_backtest(equity, trades)
    click.echo(
        "Backtest complete: "
        f"net_sharpe={metrics['net_sharpe']:.3f}, "
        f"gross_sharpe={metrics['gross_sharpe']:.3f}, "
        f"max_drawdown={metrics['max_drawdown']:.3f}, "
        f"costs={metrics['total_costs']:.6f} "
        f"(fee={metrics['total_fee_costs']:.6f}, "
        f"slippage={metrics['total_slippage_costs']:.6f})"
    )
