"""News ingestion CLI commands."""

from __future__ import annotations

from pathlib import Path

import click

from bitbat.cli._helpers import (
    _config,
    _data_path,
    _news_backend,
    _parse_datetime,
    _resolve_news_source,
)


@click.group(help="News ingestion.")
def news() -> None:
    """News command namespace."""


@news.command("pull")
@click.option("--from", "from_dt", required=True, help="Start datetime (ISO8601).")
@click.option("--to", "to_dt", required=True, help="End datetime (ISO8601).")
@click.option(
    "--source",
    type=click.Choice(["cryptocompare", "gdelt"], case_sensitive=False),
    default=None,
    help="News source override (defaults to config news_source).",
)
@click.option(
    "--output",
    type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Output directory for news parquet.",
)
def news_pull(from_dt: str, to_dt: str, source: str | None, output: Path | None) -> None:
    """Fetch historical news for feature training."""
    start = _parse_datetime(from_dt, "--from")
    end = _parse_datetime(to_dt, "--to")
    source_name = _resolve_news_source(source)
    backend = _news_backend(source_name)
    out_root = output.expanduser() if output else _data_path("raw", "news", f"{source_name}_1h")

    throttle_seconds = float(_config().get("news_throttle_seconds", 10.0))
    retry_limit = int(_config().get("news_retry_limit", 30))
    frame = backend.fetch(
        start,
        end,
        output_root=out_root,
        throttle_seconds=throttle_seconds,
        retry_limit=retry_limit,
    )
    target_path = backend._target_path(out_root)
    click.echo(f"Pulled {len(frame)} {source_name} news rows into {target_path}")
