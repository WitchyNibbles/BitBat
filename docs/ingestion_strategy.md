# Real-Time Data Ingestion Strategy

## Overview

Continuous ingestion system that:
1. Fetches BTC prices every hour from Yahoo Finance (free, unlimited)
2. Fetches crypto news from multiple free sources
3. Respects API rate limits
4. Handles failures gracefully
5. Deduplicates data

---

## Price Ingestion

### Source: Yahoo Finance (yfinance)

**Frequency:** Every hour
**Cost:** Free, unlimited
**Data:** OHLCV bars at 1-hour intervals

**Strategy:**
```python
# Every hour:
# 1. Get last ingested timestamp from database
# 2. Fetch bars from last_timestamp to now
# 3. Apply price contract
# 4. Append to parquet files
# 5. Log success/failure
```

**Error Handling:**
- Network errors: Retry 3 times with exponential backoff
- API errors: Log and skip, retry next hour
- Data validation errors: Log and alert

**Storage:**
- Location: `data/raw/prices/`
- Format: Partitioned parquet by date (`date=YYYY-MM-DD/`)
- Deduplication: By timestamp_utc

---

## News Ingestion

### Source 1: NewsAPI Free Tier

**Limits:** 100 requests/day
**Cost:** $0/month
**API Key:** Required (free at newsapi.org)
**Strategy:** Fetch every 6 hours (4 requests/day)

```python
# Every 6 hours:
# - Query: "bitcoin OR cryptocurrency"
# - Language: English
# - Sort by: publishedAt
# - Get 10 most recent articles
```

### Source 2: CryptoCompare News

**Limits:** Unlimited for news endpoint
**Cost:** $0/month
**API Key:** Not required
**Strategy:** Fetch every hour

```python
# Every hour:
# - GET https://min-api.cryptocompare.com/data/v2/news/
# - Filter by relevance to Bitcoin
# - Extract title, URL, timestamp
```

### Source 3: Reddit (PRAW) — Optional

**Limits:** 60 requests/minute
**Cost:** $0/month
**API Key:** Free app registration at reddit.com/prefs/apps
**Strategy:** Fetch every 2 hours

```python
# Every 2 hours:
# - Fetch r/Bitcoin hot posts (15 posts)
# - Fetch r/CryptoCurrency hot posts (15 posts)
# - Extract title, score, comments count
# - Calculate sentiment with VADER
```

**News Deduplication:**
- Key: URL (for NewsAPI, CryptoCompare)
- Key: Post permalink (for Reddit)

**News Storage:**
- Location: `data/raw/news/`
- Format: Partitioned parquet by date (`date=YYYY-MM-DD/`)
- Contract: `ensure_news_contract()`

---

## Scheduling

### Option 1: Python Scheduler (built-in loop)

```python
# scripts/run_ingestion_service.py
# Price: Every hour
# News: Every hour (CryptoCompare is unlimited)
# Runs as a blocking loop with graceful shutdown via SIGINT/SIGTERM
```

### Option 2: Cron (Linux/Mac)

```bash
# Price ingestion: Every hour
0 * * * * cd /opt/bitbat && poetry run python scripts/run_ingestion_service.py

# Or use CLI commands:
0 * * * * cd /opt/bitbat && poetry run bitbat ingest prices-once
0 */2 * * * cd /opt/bitbat && poetry run bitbat ingest news-once
```

### Option 3: Systemd Timers (Linux)

```ini
# /etc/systemd/system/bitbat-ingest.timer
[Timer]
OnCalendar=hourly
Persistent=true

[Install]
WantedBy=timers.target
```

### Option 4: Docker Compose

```bash
docker-compose up -d bitbat-ingest
```

---

## Rate Limiting

**NewsAPI:** 100/day limit
- Tracked in `data/newsapi_rate_limit.json`
- Before fetching, check: `requests_today < 100`
- Reset counter at midnight UTC

**CryptoCompare:** No limit on news endpoint
- No rate limiting needed

**Reddit:** 60/minute limit
- PRAW handles rate limiting automatically
- Fetch ≤15 posts per subreddit per call

---

## Error Recovery

**Network Failures:**
```python
def fetch_with_retry(func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
```

**API Errors:**
- Log error with full context
- Continue to next source
- Don't crash entire ingestion pipeline

**Data Validation Errors:**
- Log invalid data
- Skip bad records
- Continue processing

---

## Monitoring

**Metrics to Track:**
- Last successful price fetch (`price_service._get_last_timestamp()`)
- Last successful news fetch
- NewsAPI requests remaining today
- Data validation error count

**Alerts:**
- CRITICAL: No price data for 3+ hours
- WARNING: News fetch failed 3 times in row
- INFO: Daily ingestion summary

Check status with:
```bash
poetry run bitbat ingest status
```

---

## Testing

**Unit tests:**
- Rate limiter logic (`tests/autonomous/test_ingestion.py`)
- Deduplication
- Error handling

**Integration tests:**
- Full ingestion cycle with live APIs
- Concurrent ingestion (prices + news)

**End-to-end test:**
```bash
poetry run pytest tests/autonomous/test_session4_complete.py -v -s
```
