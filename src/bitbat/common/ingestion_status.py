"""
Ingestion status helper — pure Python, no Streamlit dependency.

Canonical implementation — re-exported by bitbat.gui.widgets for
backward compatibility with GUI-layer callers.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


def get_ingestion_status(data_dir: Path) -> dict[str, Any]:
    """Check freshness of price and news data on disk."""
    prices_dir = data_dir / "raw" / "prices"
    news_dir = data_dir / "raw" / "news"

    def _latest_mtime(d: Path) -> datetime | None:
        if not d.exists():
            return None
        files = list(d.glob("**/*.parquet"))
        if not files:
            return None
        return datetime.fromtimestamp(max(f.stat().st_mtime for f in files))

    prices_mtime = _latest_mtime(prices_dir)
    news_mtime = _latest_mtime(news_dir)
    now = datetime.now(UTC).replace(tzinfo=None)

    def _freshness(mtime: datetime | None) -> str:
        if mtime is None:
            return "\u26aa No data"
        hours = (now - mtime).total_seconds() / 3600
        if hours < 2:
            return "\U0001f7e2 Fresh"
        if hours < 24:
            return f"\U0001f7e1 {int(hours)}h ago"
        return f"\U0001f534 {int(hours // 24)}d ago"

    return {
        "prices": _freshness(prices_mtime),
        "news": _freshness(news_mtime),
        "prices_mtime": prices_mtime,
        "news_mtime": news_mtime,
    }


__all__ = ["get_ingestion_status"]
