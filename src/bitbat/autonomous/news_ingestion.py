"""Continuous news data ingestion service.

Fetches crypto-related news from multiple free sources:

* **CryptoCompare** — unlimited, no key required.
* **NewsAPI** — 100 requests/day on the free tier; requires ``NEWSAPI_KEY``
  environment variable.
* **Reddit (PRAW)** — optional; requires ``REDDIT_CLIENT_ID`` and
  ``REDDIT_CLIENT_SECRET`` environment variables and the ``praw`` package.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from bitbat.autonomous.rate_limiter import RateLimiter
from bitbat.contracts import ensure_news_contract
from bitbat.io.fs import write_parquet

logger = logging.getLogger(__name__)


class NewsIngestionService:
    """Fetch crypto news from multiple free sources and store as parquet.

    Usage::

        service = NewsIngestionService()
        articles_saved = service.fetch_all_sources()

    Args:
        data_dir: Root data directory.  Defaults to ``data/``.
    """

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        if data_dir is None:
            data_dir = Path("data")

        self.news_dir = data_dir / "raw" / "news"
        self.news_dir.mkdir(parents=True, exist_ok=True)

        # Optional API keys from environment.
        self.newsapi_key: Optional[str] = os.environ.get("NEWSAPI_KEY")
        self.reddit_client_id: Optional[str] = os.environ.get("REDDIT_CLIENT_ID")
        self.reddit_client_secret: Optional[str] = os.environ.get("REDDIT_CLIENT_SECRET")

        # Rate limiter for NewsAPI (100 requests/day on free tier).
        self.newsapi_limiter = RateLimiter(
            "newsapi",
            limit=100,
            period="day",
            state_file=data_dir / "newsapi_rate_limit.json",
        )

        self._sentiment = SentimentIntensityAnalyzer()
        logger.info("NewsIngestionService initialised")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sentiment_score(self, text: str) -> float:
        """Return the VADER compound sentiment score for *text*."""
        return float(self._sentiment.polarity_scores(text)["compound"])

    @staticmethod
    def _to_tz_naive(ts: object) -> datetime:
        """Convert a scalar timestamp to a tz-naive UTC datetime."""
        converted = pd.to_datetime(ts, utc=True)
        if hasattr(converted, "tz_localize"):
            return converted.tz_localize(None).to_pydatetime()
        return converted.replace(tzinfo=None)

    def _write_date_partition(self, df: pd.DataFrame) -> None:
        """Persist articles grouped by calendar date, deduplicating by URL."""
        df["_date"] = pd.to_datetime(df["published_utc"]).dt.date

        for date, group in df.groupby("_date"):
            partition_dir = self.news_dir / f"date={date}"
            partition_dir.mkdir(parents=True, exist_ok=True)
            output_file = partition_dir / "news.parquet"
            chunk = group.drop(columns=["_date"]).reset_index(drop=True)

            if output_file.exists():
                existing = pd.read_parquet(output_file)
                combined = pd.concat([existing, chunk], ignore_index=True)
                combined = combined.drop_duplicates(subset=["url"], keep="last")
                combined = combined.sort_values("published_utc").reset_index(drop=True)
                write_parquet(combined, output_file)
            else:
                write_parquet(chunk, output_file)

    # ------------------------------------------------------------------
    # Source fetchers
    # ------------------------------------------------------------------

    def fetch_cryptocompare(self, max_results: int = 20) -> list[dict]:
        """Fetch news from CryptoCompare (free, no API key needed).

        Args:
            max_results: Maximum articles to return.

        Returns:
            List of article dicts conforming to the news contract.
        """
        url = "https://min-api.cryptocompare.com/data/v2/news/"
        params: dict = {"lang": "EN"}

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            logger.error("CryptoCompare fetch failed: %s", exc)
            return []

        articles: list[dict] = []
        for item in data.get("Data", [])[:max_results]:
            try:
                published = pd.to_datetime(item["published_on"], unit="s").to_pydatetime()
                articles.append(
                    {
                        "published_utc": published,
                        "title": str(item.get("title", "")),
                        "url": str(item.get("url", "")),
                        "source": str(item.get("source", "cryptocompare")),
                        "lang": "en",
                        "sentiment_score": self._sentiment_score(str(item.get("title", ""))),
                    }
                )
            except Exception as exc:
                logger.debug("Skipping malformed CryptoCompare item: %s", exc)

        logger.info("Fetched %d articles from CryptoCompare", len(articles))
        return articles

    def fetch_newsapi(self, max_results: int = 10) -> list[dict]:
        """Fetch from NewsAPI (requires ``NEWSAPI_KEY``; 100 requests/day).

        If the key is missing or the rate limit is exhausted the method
        returns an empty list without raising.

        Args:
            max_results: Maximum articles to request.

        Returns:
            List of article dicts conforming to the news contract.
        """
        if not self.newsapi_key:
            logger.warning("NEWSAPI_KEY not set — skipping NewsAPI")
            return []

        if not self.newsapi_limiter.can_make_request():
            logger.warning(
                "NewsAPI rate limit reached: %s", self.newsapi_limiter.get_status()
            )
            return []

        url = "https://newsapi.org/v2/everything"
        params: dict = {
            "q": "bitcoin OR cryptocurrency",
            "language": "en",
            "sortBy": "publishedAt",
            "apiKey": self.newsapi_key,
            "pageSize": max_results,
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            logger.error("NewsAPI fetch failed: %s", exc)
            return []

        self.newsapi_limiter.record_request()

        articles: list[dict] = []
        for item in data.get("articles", []):
            try:
                raw_ts = pd.to_datetime(item["publishedAt"], utc=True)
                published = raw_ts.tz_localize(None).to_pydatetime()
                articles.append(
                    {
                        "published_utc": published,
                        "title": str(item.get("title", "")),
                        "url": str(item.get("url", "")),
                        "source": str((item.get("source") or {}).get("name", "newsapi")),
                        "lang": "en",
                        "sentiment_score": self._sentiment_score(str(item.get("title", ""))),
                    }
                )
            except Exception as exc:
                logger.debug("Skipping malformed NewsAPI item: %s", exc)

        logger.info("Fetched %d articles from NewsAPI", len(articles))
        return articles

    def fetch_reddit(self) -> list[dict]:
        """Fetch hot posts from Bitcoin-related subreddits via PRAW (optional).

        Requires ``praw`` package and ``REDDIT_CLIENT_ID`` /
        ``REDDIT_CLIENT_SECRET`` environment variables.  Returns an empty
        list if any prerequisite is missing.

        Returns:
            List of article dicts conforming to the news contract.
        """
        try:
            import praw  # type: ignore[import-untyped]
        except ImportError:
            logger.debug("praw not installed — skipping Reddit source")
            return []

        if not self.reddit_client_id or not self.reddit_client_secret:
            logger.warning("Reddit credentials not set — skipping Reddit source")
            return []

        try:
            reddit = praw.Reddit(
                client_id=self.reddit_client_id,
                client_secret=self.reddit_client_secret,
                user_agent="bitbat_news_ingestion/1.0",
            )

            articles: list[dict] = []
            for subreddit_name in ("Bitcoin", "CryptoCurrency"):
                sub = reddit.subreddit(subreddit_name)
                for post in sub.hot(limit=15):
                    published = pd.to_datetime(post.created_utc, unit="s").to_pydatetime()
                    articles.append(
                        {
                            "published_utc": published,
                            "title": str(post.title),
                            "url": f"https://reddit.com{post.permalink}",
                            "source": f"reddit_{subreddit_name}",
                            "lang": "en",
                            "sentiment_score": self._sentiment_score(str(post.title)),
                        }
                    )

            logger.info("Fetched %d posts from Reddit", len(articles))
            return articles

        except Exception as exc:
            logger.error("Reddit fetch failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Aggregate fetch
    # ------------------------------------------------------------------

    def fetch_all_sources(self) -> int:
        """Fetch news from all available sources and persist the results.

        Returns:
            Number of unique articles saved.
        """
        all_articles: list[dict] = []
        all_articles.extend(self.fetch_cryptocompare(max_results=20))
        all_articles.extend(self.fetch_newsapi(max_results=10))
        all_articles.extend(self.fetch_reddit())

        if not all_articles:
            logger.warning("No articles fetched from any source")
            return 0

        df = pd.DataFrame(all_articles)

        # Deduplicate by URL before applying the contract.
        df = df.drop_duplicates(subset=["url"], keep="first")

        # Validate and normalise via contract.
        df = ensure_news_contract(df)

        if df.empty:
            return 0

        self._write_date_partition(df)
        logger.info("Saved %d news articles", len(df))
        return len(df)
