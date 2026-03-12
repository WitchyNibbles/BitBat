"""Data ingestion CLI commands."""

from __future__ import annotations

from pathlib import Path

import click

from bitbat.cli._helpers import _config


@click.group(help="Data ingestion commands.")
def ingest() -> None:
    """Ingestion command namespace."""


@ingest.command("prices-once")
@click.option("--symbol", default="BTC-USD", show_default=True, help="Yahoo Finance ticker.")
@click.option(
    "--interval",
    default="1h",
    show_default=True,
    help="Bar interval (e.g. '1h', '1d').",
)
def ingest_prices_once(symbol: str, interval: str) -> None:
    """Fetch the latest prices once and store them."""
    from bitbat.autonomous.price_ingestion import PriceIngestionService

    service = PriceIngestionService(symbol=symbol, interval=interval)
    count = service.fetch_with_retry()
    click.echo(f"Ingested {count} price bars")


@ingest.command("news-once")
def ingest_news_once() -> None:
    """Fetch the latest news from all sources once and store them."""
    from bitbat.autonomous.news_ingestion import NewsIngestionService

    service = NewsIngestionService()
    count = service.fetch_all_sources()
    click.echo(f"Ingested {count} news articles")


@ingest.command("macro-once")
def ingest_macro_once() -> None:
    """Fetch the latest FRED macro data once and store it."""
    from bitbat.autonomous.macro_ingestion import MacroIngestionService

    data_dir = Path(_config()["data_dir"]).expanduser()
    service = MacroIngestionService(data_dir=data_dir)
    count = service.fetch_with_retry()
    click.echo(f"Ingested {count} macro rows")


@ingest.command("onchain-once")
def ingest_onchain_once() -> None:
    """Fetch the latest on-chain data once and store it."""
    from bitbat.autonomous.onchain_ingestion import OnchainIngestionService

    data_dir = Path(_config()["data_dir"]).expanduser()
    service = OnchainIngestionService(data_dir=data_dir)
    count = service.fetch_with_retry()
    click.echo(f"Ingested {count} on-chain rows")


@ingest.command("status")
def ingest_status() -> None:
    """Show ingestion data and rate-limit status."""
    from bitbat.autonomous.news_ingestion import NewsIngestionService
    from bitbat.autonomous.price_ingestion import PriceIngestionService

    price_service = PriceIngestionService()
    last_price_ts = price_service._get_last_timestamp()

    click.echo("Ingestion Status\n")
    click.echo("Prices:")
    click.echo(f"  Last update : {last_price_ts or 'never'}")
    click.echo(f"  Data dir    : {price_service.prices_dir}")

    news_service = NewsIngestionService()
    rate_status = news_service.newsapi_limiter.get_status()

    click.echo("\nNews APIs:")
    click.echo(f"  NewsAPI key  : {'set' if news_service.newsapi_key else 'not set'}")
    click.echo(
        f"  NewsAPI usage: "
        f"{rate_status['requests_made']}/{rate_status['limit']} "
        f"({rate_status['requests_remaining']} remaining)"
    )
    reset = rate_status.get("time_until_reset")
    click.echo(f"  Reset in     : {reset or 'N/A'}")
    click.echo(f"  Reddit keys  : {'set' if news_service.reddit_client_id else 'not set'}")
    click.echo(f"  News dir     : {news_service.news_dir}")
