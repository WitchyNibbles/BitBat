"""Feature generation CLI commands."""

from __future__ import annotations

from pathlib import Path

import click
import pandas as pd

from bitbat import __version__
from bitbat.cli._helpers import (
    _config,
    _data_path,
    _ensure_path_exists,
    _news_backend,
    _resolve_news_source,
    _resolve_setting,
    _sentiment_enabled,
)
from bitbat.dataset.build import build_xy


@click.group(help="Feature generation.")
def features() -> None:
    """Features command namespace."""


@features.command("build")
@click.option("--start", default=None, help="Start datetime for feature build.")
@click.option("--end", default=None, help="End datetime for feature build.")
@click.option(
    "--label-mode",
    type=click.Choice(["return_direction", "triple_barrier"], case_sensitive=False),
    default=None,
    help="Target labeling mode for dataset generation.",
)
@click.option(
    "--take-profit",
    type=float,
    default=None,
    help="Take-profit threshold for `--label-mode triple_barrier`.",
)
@click.option(
    "--stop-loss",
    type=float,
    default=None,
    help="Stop-loss threshold for `--label-mode triple_barrier`.",
)
def features_build(
    start: str | None,
    end: str | None,
    label_mode: str | None,
    take_profit: float | None,
    stop_loss: float | None,
) -> None:
    """Build feature matrix and labels."""
    freq = _resolve_setting(None, "freq")
    horizon = _resolve_setting(None, "horizon")
    enable_sentiment = _sentiment_enabled()
    configured_mode = str(_config().get("label_mode", "return_direction"))
    resolved_label_mode = (label_mode or configured_mode).strip().lower()
    if resolved_label_mode not in {"return_direction", "triple_barrier"}:
        raise click.ClickException(
            f"Unsupported label mode '{resolved_label_mode}'. "
            "Expected 'return_direction' or 'triple_barrier'."
        )
    if resolved_label_mode != "triple_barrier" and (
        take_profit is not None or stop_loss is not None
    ):
        raise click.ClickException(
            "--take-profit/--stop-loss can only be used with --label-mode triple_barrier."
        )

    prices_path = _data_path("raw", "prices", f"btcusd_yf_{freq}.parquet")
    _ensure_path_exists(prices_path, "Prices parquet")
    news_path = None
    if enable_sentiment:
        source = _resolve_news_source()
        backend = _news_backend(source)
        news_path = backend._target_path(_data_path("raw", "news", f"{source}_1h"))
        _ensure_path_exists(news_path, "News parquet")

    if start is None or end is None:
        sample = pd.read_parquet(prices_path, columns=["timestamp_utc"])
        timestamps = pd.to_datetime(sample["timestamp_utc"])
        default_start = timestamps.min().isoformat()
        default_end = timestamps.max().isoformat()
    else:
        default_start = start
        default_end = end

    enable_garch = bool(_config().get("enable_garch", False))
    enable_macro = bool(_config().get("enable_macro", False))
    enable_onchain = bool(_config().get("enable_onchain", False))

    macro_path = None
    if enable_macro:
        macro_candidate = _data_path("raw", "macro", "fred.parquet")
        if macro_candidate.exists():
            macro_path = macro_candidate

    onchain_path = None
    if enable_onchain:
        onchain_candidate = _data_path("raw", "onchain", "blockchain_info.parquet")
        if onchain_candidate.exists():
            onchain_path = onchain_candidate

    tau_config = _config().get("tau")
    tau_value = float(tau_config) if tau_config is not None else None
    barrier_tp: float | None = None
    barrier_sl: float | None = None
    if resolved_label_mode == "triple_barrier":
        default_tp = float(_config().get("triple_barrier_take_profit", tau_value or 0.01))
        barrier_tp = float(take_profit) if take_profit is not None else default_tp
        default_sl = float(_config().get("triple_barrier_stop_loss", barrier_tp))
        barrier_sl = float(stop_loss) if stop_loss is not None else default_sl

    X, y, _meta = build_xy(
        prices_path,
        news_path,
        freq=freq,
        horizon=horizon,
        start=default_start,
        end=default_end,
        tau=tau_value,
        enable_sentiment=enable_sentiment,
        enable_garch=enable_garch,
        macro_parquet=macro_path,
        onchain_parquet=onchain_path,
        label_mode=resolved_label_mode,
        barrier_take_profit=barrier_tp,
        barrier_stop_loss=barrier_sl,
        output_root=Path(_config()["data_dir"]).expanduser(),
        seed=int(_config().get("seed", 0)),
        version=__version__,
    )
    click.echo(f"Built feature matrix with {len(X)} rows.")
