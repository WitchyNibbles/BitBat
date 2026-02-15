#!/usr/bin/env python
"""Run continuous data ingestion (prices + news).

Fetches BTC prices from Yahoo Finance and crypto news from all configured
sources once per hour, then sleeps until the next cycle.  Responds to
SIGINT / SIGTERM for graceful shutdown.

Usage::

    poetry run python scripts/run_ingestion_service.py
"""

from __future__ import annotations

import logging
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

# Allow running directly without installation.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bitbat.autonomous.news_ingestion import NewsIngestionService
from bitbat.autonomous.price_ingestion import PriceIngestionService

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOG_DIR = Path("logs")
_LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(_LOG_DIR / "ingestion.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------

_running = True


def _handle_signal(sig: int, _frame: object) -> None:
    global _running
    logger.info("Shutdown signal %s received — stopping after current cycle", sig)
    _running = False


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)

# ---------------------------------------------------------------------------
# Ingestion cycle
# ---------------------------------------------------------------------------

_INTERVAL_SECONDS = 3_600  # 1 hour


def run_ingestion_cycle() -> None:
    """Execute one complete ingestion cycle (prices + news)."""
    logger.info("=== Starting ingestion cycle ===")

    # Price ingestion.
    try:
        price_service = PriceIngestionService(symbol="BTC-USD", interval="1h")
        price_count = price_service.fetch_with_retry()
        logger.info("Prices: ingested %d bars", price_count)
    except Exception as exc:
        logger.error("Price ingestion failed: %s", exc)

    # News ingestion.
    try:
        news_service = NewsIngestionService()
        news_count = news_service.fetch_all_sources()
        logger.info("News: ingested %d articles", news_count)
    except Exception as exc:
        logger.error("News ingestion failed: %s", exc)

    logger.info("=== Ingestion cycle complete ===\n")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main() -> None:
    logger.info("Starting continuous ingestion service (interval=%ds)", _INTERVAL_SECONDS)
    logger.info("Press Ctrl+C to stop")

    # Run one cycle immediately on start.
    run_ingestion_cycle()

    while _running:
        next_run = time.monotonic() + _INTERVAL_SECONDS
        next_run_dt = datetime.fromtimestamp(time.time() + _INTERVAL_SECONDS)
        logger.info(
            "Next ingestion at %s (in %.0f minutes)",
            next_run_dt.strftime("%Y-%m-%d %H:%M:%S"),
            _INTERVAL_SECONDS / 60,
        )

        # Sleep in 10-second chunks to allow timely shutdown.
        while _running and time.monotonic() < next_run:
            time.sleep(10)

        if _running:
            run_ingestion_cycle()

    logger.info("Ingestion service stopped")


if __name__ == "__main__":
    main()
