# Architecture

**Analysis Date:** 2026-02-24

## Pattern Overview

**Overall:** Layered pipeline architecture with strict contract-based data validation

**Key Characteristics:**
- Schema contracts enforced at every pipeline stage via `bitbat/contracts.py` — violations raise `ContractError` and halt execution
- Walk-forward time-series cross-validation with embargo bars prevents data leakage
- Modular feature engineering with pluggable feature sources (sentiment, GARCH, macro, on-chain)
- XGBoost regression as the core prediction model with ensemble capabilities
- Persistent autonomous monitoring agent with state tracking via SQLite

## Layers

**Ingestion Layer:**
- Purpose: Fetch external data from multiple sources and persist as Parquet
- Location: `src/bitbat/ingest/`
- Contains: Price fetchers (`prices.py`), news aggregators (`news_gdelt.py`, `news_cryptocompare.py`), macro data (`macro_fred.py`), on-chain metrics (`onchain.py`)
- Depends on: External APIs (yfinance, GDELT, CryptoCompare, FRED, blockchain.info)
- Used by: Feature engineering and autonomous ingestion modules

**Time-Alignment & Calendar Layer:**
- Purpose: Enforce UTC normalization, embargo bars, and walk-forward split logic to prevent leakage
- Location: `src/bitbat/timealign/`
- Contains: Calendar utilities (`calendar.py`), bucketing logic (`bucket.py`), purging rules (`purging.py`)
- Depends on: Ingestion layer data
- Used by: Dataset builder, model evaluation

**Feature Engineering Layer:**
- Purpose: Transform raw data into ML-ready features
- Location: `src/bitbat/features/`
- Contains: Price features (`price.py` — ATR, MACD, RSI, OBV, lagged returns), sentiment (`sentiment.py`), volatility (`volatility.py` — GARCH), macro indicators (`macro.py`), on-chain signals (`onchain.py`)
- Depends on: Ingestion layer, time-alignment utilities
- Used by: Dataset assembly

**Labeling Layer:**
- Purpose: Compute forward returns and generate 3-class labels (up/down/flat) from price targets
- Location: `src/bitbat/labeling/`
- Contains: Forward return computation (`returns.py`), threshold-based classification
- Depends on: Prices, configured label threshold (tau)
- Used by: Dataset builder

**Dataset Assembly Layer:**
- Purpose: Merge features and labels; enforce walk-forward splits with embargo bars
- Location: `src/bitbat/dataset/`
- Contains: Dataset builder (`build.py` with `build_xy` and feature generation), split logic (`splits.py`), metadata (`DatasetMeta` dataclass)
- Depends on: Features, labels, time-alignment
- Used by: Model training, backtesting

**Model Layer:**
- Purpose: Train, persist, evaluate, and serve XGBoost predictions
- Location: `src/bitbat/model/`
- Contains: Training (`train.py`), inference (`infer.py`), evaluation (`evaluate.py`), optimization (`optimize.py`), walk-forward CV (`walk_forward.py`), ensemble (`ensemble.py`), persistence (`persist.py`)
- Depends on: Dataset layer
- Used by: Batch prediction, API, autonomous agent

**Backtest Layer:**
- Purpose: Convert model probabilities into trading signals and measure strategy performance (Sharpe, max drawdown, hit rate)
- Location: `src/bitbat/backtest/`
- Contains: Strategy execution engine (`engine.py`), metrics computation (`metrics.py`)
- Depends on: Model predictions
- Used by: Model evaluation, analytics

**Analytics Layer:**
- Purpose: Feature importance, explainability, Monte Carlo simulations, backtest reports
- Location: `src/bitbat/analytics/`
- Contains: Feature analysis (`feature_analysis.py`), SHAP explainer (`explainer.py`), Monte Carlo (`monte_carlo.py`), backtest reporting (`backtest_report.py`)
- Depends on: Model, backtest results
- Used by: Dashboard, reports

**Autonomous Agent Layer:**
- Purpose: Persistent monitoring service with drift detection, automatic retraining, and multi-channel alerting
- Location: `src/bitbat/autonomous/`
- Contains: Main agent loop (`agent.py`), orchestrator (`orchestrator.py`), drift detection (`drift.py`), metrics tracking (`metrics.py`), retrainer (`retrainer.py`), ingestion tasks (`price_ingestion.py`, `macro_ingestion.py`, `onchain_ingestion.py`), batch predictor (`predictor.py`), database models (`models.py`), rate limiting (`rate_limiter.py`)
- Depends on: All layers — orchestrates entire pipeline
- Used by: CLI monitor command

**API Layer:**
- Purpose: REST endpoints for predictions, analytics, metrics, and health checks
- Location: `src/bitbat/api/`
- Contains: FastAPI app factory (`app.py`), route handlers (`routes/` — `health.py`, `predictions.py`, `analytics.py`, `metrics.py`), request/response schemas (`schemas.py`)
- Depends on: Model, analytics, autonomous metrics
- Used by: External clients, dashboard

**CLI/Orchestration Layer:**
- Purpose: Command-line interface orchestrating the entire pipeline — 9 command groups
- Location: `src/bitbat/cli.py`
- Contains: Click command handlers for prices, news, features, model, backtest, batch, monitor, validate, ingest
- Depends on: All lower layers
- Used by: End users, Docker containers

## Data Flow

**Training Pipeline:**

1. **Ingestion**: CLI invokes `bitbat prices ingest` → `prices.py` fetches OHLCV from yfinance → parquet saved to `data/raw/prices/`
2. **News Aggregation**: `bitbat news ingest` → news source (GDELT/CryptoCompare) → VADER sentiment scoring → `data/raw/news/{source}_1h/`
3. **Macro & On-Chain**: `bitbat features ingest-macro`, `bitbat features ingest-onchain` → external APIs → parquet storage
4. **Feature Engineering**: `bitbat features generate` → combines price, sentiment, macro, on-chain features via `dataset/build.py:_generate_price_features()` → temporary feature DataFrames
5. **Dataset Assembly**: `bitbat dataset build` → calls `build_xy()` → merges features + forward returns + labels → enforces walk-forward splits with embargo bars → contracts validated → `data/features/{freq}_{horizon}/dataset.parquet` + `meta.json`
6. **Model Training**: `bitbat model train` → calls `fit_xgb()` → XGBoost trained on training fold → persisted to `models/{freq}_{horizon}/xgb.json` → metrics to `metrics/cv_summary.json`
7. **Backtesting**: `bitbat backtest run` → loads model + test data → runs strategy → computes Sharpe, drawdown, hit rate

**Prediction Pipeline:**

1. **Batch Prediction**: `bitbat batch predict` → loads latest model → predicts next bar → saves to `data/predictions/{freq}_{horizon}.parquet`
2. **Continuous Monitoring**: `bitbat monitor start` → autonomous agent ingests fresh data hourly → runs batch prediction → checks drift metrics (hit-rate, Sharpe) against thresholds → triggers retraining if drift detected → stores state in `data/autonomous.db`
3. **API Serving**: FastAPI endpoint `/predictions/{freq}/{horizon}` → loads model from disk → infers on latest data → returns probabilities + confidence

**State Management:**

- Configuration: `src/bitbat/config/default.yaml` loaded via `config/loader.py` — controls `freq`, `horizon`, `tau`, feature toggles, autonomous thresholds
- Model State: Persisted via `model/persist.py` — XGBoost booster serialized to JSON
- Autonomous State: SQLite database (`data/autonomous.db`) tracks last run times, drift flags, retraining counts — enables restart resilience
- Metrics: Walk-forward CV summary in JSON — enables threshold-based retraining decisions

## Key Abstractions

**Contract Validation:**
- Purpose: Enforces schema at pipeline boundaries
- Examples: `ensure_prices_contract()`, `ensure_news_contract()`, `ensure_feature_contract()` in `src/bitbat/contracts.py`
- Pattern: Each function validates columns, types, datetime normalization — raises `ContractError` on violation

**DatasetMeta:**
- Purpose: Captures dataset statistics and reproducibility info
- Examples: `src/bitbat/dataset/build.py` — freq, horizon, row count, target distribution, seed
- Pattern: Serialized to JSON alongside dataset parquet for audit trail

**Time Series Splits:**
- Purpose: Walk-forward cross-validation preventing data leakage
- Examples: `src/bitbat/dataset/splits.py:walk_forward()` — embargo bars between train/test, purging rules for correlated bars
- Pattern: Returns list of (train_idx, test_idx) tuples; enforced in `build_xy()`

**Feature Decorators:**
- Purpose: Compositional feature generation with optional toggles
- Examples: Sentiment aggregation (`features/sentiment.py`) only computed if `enable_sentiment=True`
- Pattern: Feature functions check config, skip gracefully if disabled

**Autonomous Orchestrator:**
- Purpose: Coordinates periodic tasks (ingestion, prediction, monitoring, retraining)
- Examples: `src/bitbat/autonomous/orchestrator.py` — schedules tasks with configurable intervals
- Pattern: Each task is a separate module; orchestrator manages execution order and state

## Entry Points

**CLI:**
- Location: `src/bitbat/cli.py`
- Triggers: User runs `poetry run bitbat <command>`
- Responsibilities: Route to appropriate command handler; load config; invoke lower-layer functions; handle error messages

**FastAPI:**
- Location: `src/bitbat/api/app.py`
- Triggers: `poetry run uvicorn bitbat.api.app:app --reload`
- Responsibilities: Mount routers; serve HTTP requests; validate schemas via Pydantic; return JSON responses

**Autonomous Agent:**
- Location: `src/bitbat/autonomous/agent.py`
- Triggers: `bitbat monitor start` or Docker service `bitbat-monitor`
- Responsibilities: Initialize scheduler; spawn periodic ingestion/prediction/monitoring tasks; persist state; handle graceful shutdown

**Streamlit Dashboard:**
- Location: `streamlit/app.py` (outside src/ — referenced in CLAUDE.md)
- Triggers: `poetry run streamlit run streamlit/app.py`
- Responsibilities: Read data from `data/` directory; render interactive visualizations; display metrics

## Error Handling

**Strategy:** Contract violations halt execution; data errors logged with context

**Patterns:**
- `ContractError` raised in `src/bitbat/contracts.py` when schema is violated — caller decides to halt or skip
- CLI wraps command handlers with try/except → click.ClickException() → user-friendly error message
- Feature generation uses graceful degradation: if optional library (arch for GARCH) missing, log warning and skip feature
- Model inference catches loading errors → returns None or raises ValidationError to API

## Cross-Cutting Concerns

**Logging:** Uses Python `logging` module; loggers created per module via `__name__`; configured in CLI entry point; autonomous agent logs to file and console

**Validation:** Contract-based at pipeline boundaries; Pydantic schemas in API routes; custom validators in feature/labeling functions

**Authentication:** API endpoints do not enforce auth in base code — expected to be handled by reverse proxy (nginx in docker-compose)

**Time Zone Handling:** All timestamps normalized to UTC in `timealign/calendar.py:ensure_utc()` — ensures consistency across features, labels, predictions

**Configuration Management:** Single source of truth in `src/bitbat/config/default.yaml` — loaded once at startup via `config/loader.py` — runtime config accessible via `get_runtime_config()`

---

*Architecture analysis: 2026-02-24*
