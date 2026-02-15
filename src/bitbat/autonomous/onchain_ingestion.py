"""Autonomous on-chain data ingestion service."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from bitbat.ingest.onchain import fetch_blockchain_info

logger = logging.getLogger(__name__)


class OnchainIngestionService:
    """Incrementally fetch blockchain metrics and append to local parquet store."""

    def __init__(self, data_dir: Path | None = None) -> None:
        if data_dir is None:
            data_dir = Path("data")
        self.onchain_dir = data_dir / "raw" / "onchain"
        self.onchain_dir.mkdir(parents=True, exist_ok=True)
        logger.info("OnchainIngestionService initialised: %s", self.onchain_dir)

    def _get_last_date(self) -> datetime | None:
        """Return the latest date found in the stored parquet file."""
        parquet_file = self.onchain_dir / "blockchain_info.parquet"
        if not parquet_file.exists():
            return None
        try:
            df = pd.read_parquet(parquet_file)
            if df.empty or "date" not in df.columns:
                return None
            max_date = pd.to_datetime(df["date"]).max()
            return max_date.to_pydatetime()
        except Exception:
            logger.warning("Could not read existing onchain parquet")
            return None

    def fetch_latest(self) -> int:
        """Fetch on-chain data from the last stored date to today.

        Returns the number of new rows fetched.
        """
        last = self._get_last_date()
        start = datetime(2017, 1, 1) if last is None else last - timedelta(days=5)

        logger.info("Fetching on-chain data from %s", start.date())
        frame = fetch_blockchain_info(start=start, output_root=self.onchain_dir)
        new_rows = len(frame) if frame is not None else 0
        logger.info("Fetched %d on-chain rows", new_rows)
        return new_rows

    def fetch_with_retry(self, max_retries: int = 3) -> int:
        """Fetch with exponential backoff on failure."""
        for attempt in range(max_retries):
            try:
                return self.fetch_latest()
            except Exception as exc:
                if attempt >= max_retries - 1:
                    logger.error("All %d onchain fetch attempts failed: %s", max_retries, exc)
                    raise
                wait = 2**attempt
                logger.warning(
                    "Onchain fetch attempt %d/%d failed: %s — retrying in %ds",
                    attempt + 1,
                    max_retries,
                    exc,
                    wait,
                )
                time.sleep(wait)
        return 0  # unreachable, satisfies type checker
