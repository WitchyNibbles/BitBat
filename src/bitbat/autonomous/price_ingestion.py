"""Continuous price data ingestion service.

Fetches BTC OHLCV bars from Yahoo Finance on demand and stores them in
date-partitioned parquet files under ``data/raw/prices/``.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from bitbat.contracts import ensure_prices_contract
from bitbat.io.fs import write_parquet

logger = logging.getLogger(__name__)

# How many days to fetch on the very first run (no existing data).
_FIRST_RUN_DAYS = 7


class PriceIngestionService:
    """Continuously fetch and store BTC price bars.

    On each call to :meth:`fetch_latest` the service discovers the most
    recent timestamp already stored on disk, then downloads all bars from
    that point up to *now* and appends them (deduplicating by timestamp).

    Usage::

        service = PriceIngestionService(symbol='BTC-USD', interval='1h')
        bars_added = service.fetch_with_retry()

    Args:
        symbol: Yahoo Finance ticker symbol (e.g. ``'BTC-USD'``).
        interval: Bar interval understood by yfinance (e.g. ``'1h'``).
        data_dir: Root data directory.  Defaults to ``data/``.
    """

    def __init__(
        self,
        symbol: str = "BTC-USD",
        interval: str = "1h",
        data_dir: Path | None = None,
    ) -> None:
        self.symbol = symbol
        self.interval = interval

        if data_dir is None:
            data_dir = Path("data")

        self.prices_dir = data_dir / "raw" / "prices"
        self.prices_dir.mkdir(parents=True, exist_ok=True)

        logger.info("PriceIngestionService initialised: %s @ %s", symbol, interval)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_last_timestamp(self) -> datetime | None:
        """Return the latest ``timestamp_utc`` found across all stored parquet files."""
        parquet_files = list(self.prices_dir.glob("**/*.parquet"))
        if not parquet_files:
            return None

        max_ts: pd.Timestamp | None = None
        for path in parquet_files:
            try:
                df = pd.read_parquet(path, columns=["timestamp_utc"])
                if df.empty or "timestamp_utc" not in df.columns:
                    continue
                file_max = pd.to_datetime(df["timestamp_utc"]).max()
                if max_ts is None or file_max > max_ts:
                    max_ts = file_max
            except Exception as exc:
                logger.warning("Could not read %s: %s", path, exc)

        if max_ts is None:
            return None
        return max_ts.to_pydatetime()

    @staticmethod
    def _to_tz_naive(series: pd.Series) -> pd.Series:
        """Convert a datetime series to tz-naive UTC."""
        converted = pd.to_datetime(series, utc=True)
        return converted.dt.tz_localize(None)

    def _write_partition(self, df: pd.DataFrame, date: object) -> None:
        """Write (or merge) a single day's bars into a date partition."""
        partition_dir = self.prices_dir / f"date={date}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        output_file = (
            partition_dir / f"{self.symbol.replace('-', '').lower()}_{self.interval}.parquet"
        )

        if output_file.exists():
            existing = pd.read_parquet(output_file)
            combined = pd.concat([existing, df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["timestamp_utc"], keep="last")
            combined = combined.sort_values("timestamp_utc").reset_index(drop=True)
            write_parquet(combined, output_file)
        else:
            write_parquet(df, output_file)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fetch_latest(self) -> int:
        """Download bars since the last stored timestamp and persist them.

        Returns:
            Number of new bars written.
        """
        last_ts = self._get_last_timestamp()

        if last_ts is None:
            start = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=_FIRST_RUN_DAYS)
            logger.info("First fetch — downloading last %d days from %s", _FIRST_RUN_DAYS, start)
        else:
            # Offset by one bar to avoid re-fetching the last stored bar.
            start = last_ts + timedelta(hours=1)
            logger.info("Fetching from %s to now", start)

        end = datetime.now(UTC).replace(tzinfo=None)

        if start >= end:
            logger.info("Price data already up to date")
            return 0

        ticker = yf.Ticker(self.symbol)
        raw = ticker.history(start=start, end=end, interval=self.interval)

        if raw.empty:
            logger.warning("No data returned from yfinance for %s", self.symbol)
            return 0

        frame = raw.reset_index()

        # Normalise column names (yfinance may use 'Datetime' or 'Date').
        rename_map = {
            "Datetime": "timestamp_utc",
            "Date": "timestamp_utc",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
        frame = frame.rename(columns={k: v for k, v in rename_map.items() if k in frame.columns})

        frame["source"] = "yfinance"

        # Strip timezone so the contract can normalise.
        frame["timestamp_utc"] = self._to_tz_naive(frame["timestamp_utc"])

        # Validate via contract (raises ContractError if columns missing).
        frame = ensure_prices_contract(frame)

        if frame.empty:
            return 0

        # Write one parquet file per calendar date.
        frame["_date"] = pd.to_datetime(frame["timestamp_utc"]).dt.date
        for date, group in frame.groupby("_date"):
            self._write_partition(group.drop(columns=["_date"]), date)

        logger.info("Ingested %d new price bars for %s", len(frame), self.symbol)
        return len(frame)

    def fetch_with_retry(self, max_retries: int = 3) -> int:
        """Fetch latest bars, retrying on transient errors.

        Args:
            max_retries: Maximum number of attempts before re-raising.

        Returns:
            Number of bars written on success.
        """
        for attempt in range(max_retries):
            try:
                return self.fetch_latest()
            except Exception as exc:
                if attempt >= max_retries - 1:
                    logger.error("All %d fetch attempts failed: %s", max_retries, exc)
                    raise
                wait = 2**attempt
                logger.warning(
                    "Fetch attempt %d/%d failed: %s — retrying in %ds",
                    attempt + 1,
                    max_retries,
                    exc,
                    wait,
                )
                time.sleep(wait)
        return 0  # unreachable, but satisfies type checker
