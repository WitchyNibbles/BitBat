"""Bitcoin on-chain data ingestion from blockchain.info."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from bitbat.io.fs import write_parquet

logger = logging.getLogger(__name__)

BLOCKCHAIN_INFO_BASE = "https://api.blockchain.info/charts"

BLOCKCHAIN_METRICS: dict[str, str] = {
    "hash-rate": "hashrate",
    "n-transactions": "tx_count",
    "mempool-size": "mempool_size",
    "avg-block-size": "avg_block_size",
}


def _target_path(root: Path | str | None = None) -> Path:
    """Build the parquet target path for on-chain data."""
    base = Path(root) if root is not None else Path("data") / "raw" / "onchain"
    return base / "blockchain_info.parquet"


def _fetch_metric(
    metric: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Fetch a single blockchain.info chart metric."""
    timespan_days = (end - start).days + 1
    params: dict[str, str] = {
        "timespan": f"{timespan_days}days",
        "start": start.strftime("%Y-%m-%d"),
        "format": "json",
        "sampled": "true",
    }

    url = f"{BLOCKCHAIN_INFO_BASE}/{metric}"
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    values = data.get("values", [])
    if not values:
        return pd.DataFrame(columns=["date", "value"])

    rows = [{"date": pd.Timestamp(v["x"], unit="s"), "value": float(v["y"])} for v in values]
    return pd.DataFrame(rows)


def fetch_blockchain_info(
    start: datetime = datetime(2017, 1, 1),
    end: datetime | None = None,
    *,
    metrics: dict[str, str] | None = None,
    output_root: Path | str | None = None,
) -> pd.DataFrame:
    """Fetch Bitcoin on-chain metrics from blockchain.info and persist to parquet.

    Parameters
    ----------
    start:
        Start date for observations.
    end:
        End date (defaults to today).
    metrics:
        Mapping of blockchain.info chart name to column name.
        Defaults to ``BLOCKCHAIN_METRICS``.
    output_root:
        Root directory for output.

    Returns
    -------
    DataFrame with ``date`` index and one column per metric.
    """
    if end is None:
        end = datetime.now()
    if metrics is None:
        metrics = BLOCKCHAIN_METRICS

    merged: pd.DataFrame | None = None
    for chart_name, col_name in metrics.items():
        logger.info("Fetching blockchain.info metric %s → %s", chart_name, col_name)
        df = _fetch_metric(chart_name, start, end)
        if df.empty:
            logger.warning("No data returned for metric %s", chart_name)
            continue
        df = df.rename(columns={"value": col_name}).set_index("date")
        merged = df if merged is None else merged.join(df, how="outer")

    if merged is None or merged.empty:
        raise ValueError("No on-chain data returned from blockchain.info.")

    merged = merged.sort_index().ffill()

    target = _target_path(output_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(merged.reset_index(), target)
    logger.info("Wrote on-chain data to %s (%d rows)", target, len(merged))

    return merged
