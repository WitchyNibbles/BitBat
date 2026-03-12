"""Price data CLI commands."""

from __future__ import annotations

from pathlib import Path

import click

from bitbat.cli._helpers import _data_path, _parse_datetime, _resolve_setting
from bitbat.ingest import prices as prices_module


@click.group(help="Price data operations.")
def prices() -> None:
    """Price command namespace."""


@prices.command("pull")
@click.option("--symbol", required=True, help="Ticker symbol to download.")
@click.option("--start", required=True, help="Start date (YYYY-MM-DD).")
@click.option("--interval", default=None, help="Data interval (defaults to config freq).")
@click.option(
    "--output",
    type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Output directory for prices parquet.",
)
def prices_pull(
    symbol: str,
    start: str,
    interval: str | None,
    output: Path | None,
) -> None:
    """Pull price data from Yahoo Finance."""
    freq = interval or _resolve_setting(None, "freq")
    start_dt = _parse_datetime(start, "--start")
    out_root = output.expanduser() if output else _data_path("raw", "prices")

    frame = prices_module.fetch_yf(symbol, freq, start_dt, output_root=out_root)
    target_path = prices_module._target_path(symbol, freq, out_root)
    click.echo(f"Pulled {len(frame)} rows for {symbol} {freq} into {target_path}")
