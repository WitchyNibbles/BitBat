# Integrations

**Analysis Date:** 2026-02-24

## External Data Providers

### Yahoo Finance (Market Prices)

- Purpose: BTC OHLCV ingestion.
- Modules:
  - `src/bitbat/ingest/prices.py` (`fetch_yf`).
  - `src/bitbat/autonomous/price_ingestion.py` (`PriceIngestionService`).
- Library: `yfinance`.
- Output:
  - Batch path writes partitioned parquet under `data/raw/prices/`.
  - Autonomous service writes date-partitioned parquet under `data/raw/prices/date=YYYY-MM-DD/`.

### CryptoCompare (News)

- Purpose: crypto news headlines and metadata.
- Modules:
  - `src/bitbat/ingest/news_cryptocompare.py`.
  - `src/bitbat/autonomous/news_ingestion.py` (`fetch_cryptocompare`).
- Endpoint: `https://min-api.cryptocompare.com/data/v2/news/`.
- Behavior: retry/backoff logic in batch ingestion module; sentiment score derived with VADER.

### GDELT (News)

- Purpose: keyword-based BTC/crypto article retrieval.
- Module: `src/bitbat/ingest/news_gdelt.py`.
- Endpoint: `https://api.gdeltproject.org/api/v2/doc/doc`.
- Behavior: one-hour windows, retry with exponential backoff, optional throttle.

### FRED (Macro)

- Purpose: optional macroeconomic features.
- Module: `src/bitbat/ingest/macro_fred.py`.
- Endpoint: `https://api.stlouisfed.org/fred/series/observations`.
- Auth: optional API key via config/env (`fred_api_key` / `FRED_API_KEY`).

### blockchain.info (On-chain)

- Purpose: optional on-chain feature inputs.
- Module: `src/bitbat/ingest/onchain.py`.
- Endpoint family: `https://api.blockchain.info/charts/{metric}`.

### NewsAPI and Reddit (Autonomous News Service)

- Purpose: supplemental live news ingestion.
- Module: `src/bitbat/autonomous/news_ingestion.py`.
- Endpoints/services:
  - NewsAPI `https://newsapi.org/v2/everything`.
  - Reddit via `praw` client library.
- Credentials:
  - `NEWSAPI_KEY` for NewsAPI.
  - `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET` for Reddit.
- Rate-limit control: `RateLimiter` in `src/bitbat/autonomous/rate_limiter.py`.

## Alerting Integrations

- Module: `src/bitbat/autonomous/alerting.py`.
- Channels:
  - SMTP email (`smtplib`) using config keys under `autonomous.alerts`.
  - Discord/Slack-style webhook via HTTP POST.
  - Telegram Bot API (`https://api.telegram.org/bot{token}/sendMessage`).
- Trigger source: monitoring/retraining flow in `src/bitbat/autonomous/agent.py`.

## Internal Service-to-Service Integrations

- API -> autonomous DB:
  - Prediction and analytics endpoints query SQLite through `AutonomousDB`.
  - Files: `src/bitbat/api/routes/predictions.py`, `src/bitbat/api/routes/analytics.py`, `src/bitbat/api/routes/metrics.py`.
- Monitor agent -> predictor/validator/drift/retrainer:
  - Composed in `src/bitbat/autonomous/agent.py`.
- CLI batch inference -> autonomous DB write-through:
  - `batch run` in `src/bitbat/cli.py` stores to parquet and then DB.

## Data Storage Integrations

- Parquet storage:
  - Read/write wrappers in `src/bitbat/io/fs.py`.
  - Contracts in `src/bitbat/contracts.py` enforce schemas for prices/news/features/predictions.
- SQLite:
  - Schema and engine setup in `src/bitbat/autonomous/models.py`.
  - Repository operations in `src/bitbat/autonomous/db.py`.
- DuckDB:
  - Ad hoc in-memory query helper in `src/bitbat/io/duck.py`.

## Deployment Integrations

- Docker Compose app services: `docker-compose.yml`.
  - `bitbat-api`, `bitbat-ingest`, `bitbat-monitor`, `bitbat-ui`, optional `nginx` profile.
- Observability overlay: `deployment/docker-compose.monitoring.yml`.
  - Prometheus scrapes `/metrics` from API (`deployment/prometheus.yml`).
  - Grafana data source expected to be Prometheus container.
- Reverse proxy: `deployment/nginx.conf` routes API and Streamlit paths.

## Integration Boundaries and Contracts

- Network boundaries are primarily HTTP-based (`requests`, `yfinance`, webhook POSTs).
- Storage boundaries are file-based parquet plus single SQLite file.
- Schema boundary checks are explicit at module edges via:
  - `ensure_prices_contract`
  - `ensure_news_contract`
  - `ensure_feature_contract`
  - `ensure_predictions_contract`
  in `src/bitbat/contracts.py`.

