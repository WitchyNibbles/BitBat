"""RSS-based crypto news ingestion — free, unlimited, no API keys.

Fetches headlines from major crypto news RSS feeds (CoinDesk, CoinTelegraph,
Bitcoin Magazine, Decrypt, The Block, etc.), scores them with VADER sentiment,
and persists to the standard news parquet format.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from bitbat.ingest.http_helpers import merge_and_save_news_parquet

LOGGER = logging.getLogger(__name__)

# --- Free crypto RSS feeds (no keys, no rate limits) ---
RSS_FEEDS: list[dict[str, str]] = [
    {"name": "CoinDesk", "url": "https://www.coindesk.com/arc/outboundfeeds/rss/"},
    {"name": "CoinTelegraph", "url": "https://cointelegraph.com/rss"},
    {"name": "Bitcoin Magazine", "url": "https://bitcoinmagazine.com/.rss/full/"},
    {"name": "Decrypt", "url": "https://decrypt.co/feed"},
    {"name": "The Block", "url": "https://www.theblock.co/rss.xml"},
    {"name": "CryptoSlate", "url": "https://cryptoslate.com/feed/"},
    {"name": "NewsBTC", "url": "https://www.newsbtc.com/feed/"},
    {"name": "Bitcoinist", "url": "https://bitcoinist.com/feed/"},
]

RESULT_COLUMNS = ["published_utc", "title", "url", "source", "lang", "sentiment_score"]


def _target_path(root: Path | str | None = None) -> Path:
    base = (
        Path(root)
        if root is not None
        else Path("data") / "raw" / "news" / "rss_1h"
    )
    return base / "rss_crypto_1h.parquet"


def _parse_rss_date(date_str: str | None) -> datetime | None:
    """Parse RFC-822 / RFC-2822 date strings commonly used in RSS."""
    if not date_str:
        return None
    try:
        return parsedate_to_datetime(date_str)
    except Exception:
        pass
    # Fallback: ISO-style dates some feeds use
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except Exception:
        return None


def _fetch_feed(feed: dict[str, str], timeout: int = 15) -> list[dict[str, Any]]:
    """Fetch and parse a single RSS feed, returning article dicts."""
    name = feed["name"]
    url = feed["url"]
    articles: list[dict[str, Any]] = []

    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "BitBat/1.0"})
        resp.raise_for_status()
    except Exception as exc:
        LOGGER.warning("RSS fetch failed for %s: %s", name, exc)
        return []

    try:
        root = ET.fromstring(resp.text)  # noqa: S314
    except ET.ParseError as exc:
        LOGGER.warning("RSS XML parse failed for %s: %s", name, exc)
        return []

    # Handle both RSS 2.0 (<channel><item>) and Atom (<entry>) formats
    items = root.findall(".//item")
    if not items:
        # Atom namespace
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        items = root.findall(".//atom:entry", ns)

    for item in items:
        try:
            # RSS 2.0 fields
            title_el = item.find("title")
            link_el = item.find("link")
            pub_el = item.find("pubDate")

            # Atom fallback
            if title_el is None:
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                title_el = item.find("atom:title", ns)
            if link_el is None:
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                link_el = item.find("atom:link", ns)
                if link_el is not None and link_el.text is None:
                    # Atom <link> uses href attribute
                    link_text = link_el.get("href", "")
                else:
                    link_text = link_el.text if link_el is not None else ""
            else:
                link_text = link_el.text or ""
            if pub_el is None:
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                pub_el = item.find("atom:updated", ns) or item.find("atom:published", ns)

            title_text = (title_el.text or "").strip() if title_el is not None else ""
            link_text = (link_text or "").strip()
            pub_date = _parse_rss_date(
                (pub_el.text or "").strip() if pub_el is not None else None
            )

            if not title_text or not link_text:
                continue

            articles.append({
                "title": title_text,
                "url": link_text,
                "published_utc": pub_date,
                "source": name,
            })
        except Exception as exc:
            LOGGER.debug("Skipping malformed RSS item from %s: %s", name, exc)

    LOGGER.info("Fetched %d articles from %s RSS", len(articles), name)
    return articles


def _articles_to_frame(articles: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert raw article dicts to a scored DataFrame."""
    if not articles:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    frame = pd.DataFrame(articles)

    # Normalise timestamps
    frame["published_utc"] = pd.to_datetime(
        frame["published_utc"], utc=True, errors="coerce"
    )
    frame["published_utc"] = frame["published_utc"].dt.tz_localize(None)
    frame = frame.dropna(subset=["published_utc", "url"])
    frame = frame[frame["url"].str.startswith("http", na=False)]
    frame["title"] = frame["title"].fillna("")
    frame["source"] = frame["source"].fillna("rss")
    frame["lang"] = "en"

    # VADER sentiment scoring
    from bitbat.features.sentiment import score_vader

    frame["sentiment_score"] = score_vader(frame["title"])

    return frame[RESULT_COLUMNS]


def fetch(
    from_dt: datetime | None = None,
    to_dt: datetime | None = None,
    *,
    output_root: Path | str | None = None,
    feeds: list[dict[str, str]] | None = None,
    **_kwargs: Any,
) -> pd.DataFrame:
    """Fetch crypto news from RSS feeds and persist to parquet.

    Args:
        from_dt: Optional start filter (only keep articles after this).
        to_dt: Optional end filter (only keep articles before this).
        output_root: Override for output directory.
        feeds: Override list of RSS feeds to use.

    Returns:
        Combined DataFrame of all fetched articles.
    """
    feed_list = feeds or RSS_FEEDS
    all_articles: list[dict[str, Any]] = []

    for feed in feed_list:
        all_articles.extend(_fetch_feed(feed))

    frame = _articles_to_frame(all_articles)

    if frame.empty:
        LOGGER.warning("No articles fetched from any RSS feed")
        target_path = _target_path(output_root)
        return merge_and_save_news_parquet([], target_path, RESULT_COLUMNS)

    # Apply date filters if provided
    if from_dt is not None:
        from_naive = pd.Timestamp(from_dt).tz_localize(None)
        frame = frame[frame["published_utc"] >= from_naive]
    if to_dt is not None:
        to_naive = pd.Timestamp(to_dt).tz_localize(None)
        frame = frame[frame["published_utc"] <= to_naive]

    # Deduplicate by URL
    frame = frame.drop_duplicates(subset=["url"], keep="first")

    target_path = _target_path(output_root)
    return merge_and_save_news_parquet([frame], target_path, RESULT_COLUMNS)
