# External Integrations

**Analysis Date:** 2026-02-24

## APIs & External Services

**Price Data:**
- Yahoo Finance (yfinance)
  - What it's used for: OHLCV (open, high, low, close, volume) candle data for BTC-USD
  - SDK/Client: `yfinance` 0.2.40 package
  - Auth: None required (public API)
  - Implementation: `src/bitbat/ingest/prices.py`
  - Limitations: Intraday data restricted to ~2 years per request; chunked downloads used

**News Data:**
- GDELT (Global Database of Events, Language, and Tone)
  - What it's used for: News articles with published timestamps, titles, URLs, sources, language, sentiment
  - Endpoint: `https://api.gdeltproject.org/api/v2/doc/doc`
  - Auth: None (public API)
  - Implementation: `src/bitbat/ingest/news_gdelt.py`
  - Keywords: bitcoin, btc, crypto, cryptocurrency
  - Rate limiting: Configurable throttle (`news_throttle_seconds: 10.0` in default config)

- CryptoCompare
  - What it's used for: Alternative news source for cryptocurrency events
  - Auth: None (public API)
  - Implementation: `src/bitbat/ingest/news_cryptocompare.py`
  - Configuration: `news_source: "cryptocompare"` in config

**Macroeconomic Data:**
- FRED (Federal Reserve Economic Data)
  - What it's used for: US economic indicators (fed funds rate, 10Y treasury, USD index, VIX, inflation breakeven)
  - Endpoint: `https://api.stlouisfed.org/fred/series/observations`
  - Auth: API key (environment variable: `FRED_API_KEY` or config `fred_api_key`)
  - Default series: DFF, DGS10, DTWEXBGS, VIXCLS, T5YIE
  - Implementation: `src/bitbat/ingest/macro_fred.py`
  - Feature toggle: `enable_macro: false` (disabled by default)

**On-Chain Data:**
- blockchain.info
  - What it's used for: Bitcoin network metrics (hashrate, transaction count, mempool size, average block size)
  - Endpoint: `https://api.blockchain.info/charts`
  - Auth: None (public API)
  - Implementation: `src/bitbat/ingest/onchain.py`
  - Metrics: hash-rate, n-transactions, mempool-size, avg-block-size
  - Feature toggle: `enable_onchain: false` (disabled by default)

## Data Storage

**Databases:**
- Parquet (file-based, organized under `data/` directory)
  - Raw data: `data/raw/prices/`, `data/raw/news/{source}_1h/`, `data/raw/macro/`, `data/raw/onchain/`
  - Features: `data/features/{freq}_{horizon}/dataset.parquet`
  - Predictions: `data/predictions/{freq}_{horizon}.parquet`
  - Client: PyArrow 17.0.0
  - Query tool: DuckDB 1.1.2 (SQL queries on parquet files)
  - Implementation: `src/bitbat/io/fs.py` (read_parquet, write_parquet)

- SQLite
  - Purpose: Persistent state for autonomous monitoring agent (model performance, drift detection, retraining history)
  - Location: `data/autonomous.db`
  - Connection string: `sqlite:///data/autonomous.db` (configurable via `autonomous.database_url`)
  - ORM: SQLAlchemy 2.0.37
  - Implementation: `src/bitbat/autonomous/db.py`

**File Storage:**
- Local filesystem only
  - Models: `models/{freq}_{horizon}/xgb.json` (XGBoost model serialization)
  - Metrics: `metrics/cv_summary.json` (walk-forward cross-validation results)
  - Data configurable via `data_dir` parameter (default: `"data"`)

**Caching:**
- None (in-process; no Redis or Memcached)

## Authentication & Identity

**Auth Provider:**
- Custom/None - No centralized authentication provider
- API endpoints (`bitbat/api/routes/`) are public (no auth middleware)
- All authentication is via configuration (API keys stored in config or environment)

## Monitoring & Observability

**Error Tracking:**
- None detected (no Sentry, DataDog, or similar)

**Logs:**
- Python standard `logging` module
- Configured loggers in various modules: `bitbat.ingest`, `bitbat.autonomous`, `bitbat.model`, etc.
- Log level control via environment or application code
- Implementation: `src/bitbat/autonomous/agent.py`, `src/bitbat/autonomous/validator.py` use logging for drift detection events

**Performance Monitoring:**
- Autonomous agent tracks: hit rate, Sharpe ratio, MAE, directional accuracy, RMSE
- Persisted in SQLite for historical performance snapshots (`performance_snapshot_interval: 86400`)
- Metrics exported via API: `src/bitbat/api/routes/metrics.py`

## CI/CD & Deployment

**Hosting:**
- Not specified in codebase (flexible deployment)
- Docker Compose support: `docker-compose.yml` defines 4 services (bitbat-api, bitbat-ingest, bitbat-monitor, bitbat-ui) plus optional nginx reverse proxy

**CI Pipeline:**
- None detected in codebase (no GitHub Actions, GitLab CI, etc.)

**Deployment Options:**
- FastAPI server via uvicorn: `poetry run uvicorn bitbat.api.app:app --reload`
- Streamlit dashboard: `poetry run streamlit run streamlit/app.py`
- CLI batch processing: `poetry run bitbat <command>`
- Autonomous monitoring: `poetry run bitbat monitor start`

## Environment Configuration

**Required env vars:**
- `FRED_API_KEY` - Federal Reserve economic data (if `enable_macro: true`)
- Optional: Slack webhook (`SLACK_WEBHOOK_URL`), Discord webhook (`DISCORD_WEBHOOK_URL`), Telegram token/chat ID for alerting

**Secrets location:**
- Environment variables (`.env` files not checked in)
- Configuration file: `src/bitbat/config/default.yaml` (secrets as empty strings by default, filled at runtime)
- Alert channel credentials: SMTP (email), Slack, Discord, Telegram webhooks

## Webhooks & Callbacks

**Incoming:**
- None detected

**Outgoing:**
- Slack - Drift detection alerts via `slack_webhook_url` (disabled by default)
- Discord - Drift detection alerts via `discord_webhook_url` (disabled by default)
- Email - SMTP-based drift detection alerts (disabled by default, `email_enabled: false`)
- Telegram - Bot message alerts via `telegram_bot_token` and `telegram_chat_id` (disabled by default)
- Implementation: `src/bitbat/autonomous/agent.py` triggers alerts based on drift detection

## API Endpoints (Internal)

**REST API Routes:**
- `GET /health` - System health check (`src/bitbat/api/routes/health.py`)
- `GET /predictions/*` - Prediction results and forecasts (`src/bitbat/api/routes/predictions.py`)
- `GET /analytics/*` - Feature analysis, backtest reports (`src/bitbat/api/routes/analytics.py`)
- `GET /metrics/*` - Performance metrics, drift detection status (`src/bitbat/api/routes/metrics.py`)
- OpenAPI docs: `/docs` (Swagger UI), `/redoc` (ReDoc)

---

*Integration audit: 2026-02-24*
