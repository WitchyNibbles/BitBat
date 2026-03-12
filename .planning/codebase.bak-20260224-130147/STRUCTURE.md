# Codebase Structure

**Analysis Date:** 2026-02-24

## Directory Layout

```
ai-btc-predictor/
├── src/bitbat/                    # Main package root
│   ├── __init__.py                # Package initialization
│   ├── cli.py                     # CLI entry point (9 command groups)
│   ├── contracts.py               # Schema validation enforcement
│   ├── config/                    # Configuration management
│   │   ├── __init__.py
│   │   ├── loader.py              # Config loading and runtime access
│   │   └── default.yaml           # Default settings (freq, horizon, tau, toggles)
│   ├── ingest/                    # Data fetchers for external sources
│   │   ├── __init__.py
│   │   ├── prices.py              # yfinance OHLCV ingestion
│   │   ├── news_gdelt.py          # GDELT news aggregation
│   │   ├── news_cryptocompare.py  # CryptoCompare news with sentiment
│   │   ├── macro_fred.py          # FRED macro indicators
│   │   └── onchain.py             # blockchain.info on-chain metrics
│   ├── timealign/                 # Time series alignment and leakage prevention
│   │   ├── __init__.py
│   │   ├── calendar.py            # UTC normalization, date utilities
│   │   ├── bucket.py              # Time bucketing for alignment
│   │   └── purging.py             # Embargo bar logic for walk-forward
│   ├── features/                  # Feature engineering modules
│   │   ├── __init__.py
│   │   ├── price.py               # Price features (ATR, MACD, RSI, OBV, lags)
│   │   ├── sentiment.py           # Sentiment aggregation from news
│   │   ├── volatility.py          # GARCH and rolling volatility
│   │   ├── macro.py               # Macro indicators (interest rates, inflation)
│   │   └── onchain.py             # On-chain signals (whale addresses, transaction counts)
│   ├── labeling/                  # Label generation from prices
│   │   ├── __init__.py
│   │   └── returns.py             # Forward return computation and 3-class labels
│   ├── dataset/                   # Dataset assembly and splits
│   │   ├── __init__.py
│   │   ├── build.py               # Dataset assembly (build_xy, _generate_price_features)
│   │   ├── splits.py              # Walk-forward split logic with embargo
│   │   └── meta.py                # DatasetMeta dataclass
│   ├── model/                     # Model training, inference, evaluation
│   │   ├── __init__.py
│   │   ├── train.py               # XGBoost training (fit_xgb)
│   │   ├── infer.py               # Batch and single-bar inference
│   │   ├── evaluate.py            # Cross-validation and regression metrics
│   │   ├── optimize.py            # Hyperparameter optimization
│   │   ├── walk_forward.py        # Walk-forward CV orchestration
│   │   ├── ensemble.py            # Ensemble methods
│   │   └── persist.py             # Model serialization/deserialization
│   ├── backtest/                  # Strategy backtesting engine
│   │   ├── __init__.py
│   │   ├── engine.py              # Position sizing, trade execution
│   │   └── metrics.py             # Sharpe ratio, max drawdown, hit rate
│   ├── analytics/                 # Analysis and explainability
│   │   ├── __init__.py
│   │   ├── feature_analysis.py    # Feature importance and correlation
│   │   ├── explainer.py           # SHAP-based explainability
│   │   ├── monte_carlo.py         # Monte Carlo simulations
│   │   └── backtest_report.py     # Backtest summary reports
│   ├── io/                        # I/O utilities
│   │   ├── __init__.py
│   │   ├── fs.py                  # File system operations
│   │   └── duck.py                # DuckDB SQL helpers
│   ├── api/                       # FastAPI REST API
│   │   ├── __init__.py
│   │   ├── app.py                 # FastAPI app factory
│   │   ├── schemas.py             # Pydantic request/response schemas
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── health.py          # GET /health
│   │       ├── predictions.py     # GET /predictions/{freq}/{horizon}
│   │       ├── analytics.py       # GET /analytics/*
│   │       └── metrics.py         # GET /metrics/*
│   ├── autonomous/                # Autonomous monitoring agent
│   │   ├── __init__.py
│   │   ├── agent.py               # Main agent loop and scheduler
│   │   ├── orchestrator.py        # Task coordination
│   │   ├── drift.py               # Drift detection (hit-rate, Sharpe)
│   │   ├── metrics.py             # Performance metrics tracking
│   │   ├── retrainer.py           # Automatic retraining logic
│   │   ├── validator.py           # Prediction validation
│   │   ├── predictor.py           # Batch prediction service
│   │   ├── price_ingestion.py     # Periodic price fetch
│   │   ├── macro_ingestion.py     # Periodic macro data fetch
│   │   ├── onchain_ingestion.py   # Periodic on-chain data fetch
│   │   ├── models.py              # SQLAlchemy models for autonomous.db
│   │   ├── db.py                  # Database initialization and access
│   │   ├── rate_limiter.py        # API rate limiting utility
│   │   ├── continuous_trainer.py  # Continuous training orchestrator
│   │   └── alerts/                # Multi-channel alerting (if present)
│   └── gui/                       # Streamlit dashboard components
│       ├── __init__.py
│       └── widgets.py             # Reusable UI components
├── tests/                         # Test suite (54 tests)
│   ├── conftest.py                # Pytest fixtures and session config
│   ├── contracts/
│   │   ├── __init__.py
│   │   └── test_contracts.py      # Schema validation tests
│   ├── features/
│   │   ├── __init__.py
│   │   ├── test_leakage.py        # Future data leakage detection
│   │   └── test_*.py              # Feature-specific tests
│   ├── dataset/
│   │   ├── __init__.py
│   │   ├── test_build_xy.py       # Dataset assembly tests
│   │   └── test_*.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── test_train.py          # Training tests
│   │   ├── test_infer.py          # Inference tests
│   │   ├── test_evaluate.py       # Evaluation tests
│   │   ├── test_ensemble.py       # Ensemble tests
│   │   ├── test_optimize.py       # Optimization tests
│   │   ├── test_walk_forward.py   # Walk-forward CV tests
│   │   └── test_*.py
│   ├── backtest/
│   │   ├── __init__.py
│   │   ├── test_engine.py         # Backtest engine tests
│   │   └── test_*.py
│   ├── autonomous/
│   │   ├── __init__.py
│   │   ├── test_metrics.py        # Autonomous metrics tests
│   │   └── test_*.py
│   ├── timealign/
│   │   ├── __init__.py
│   │   ├── test_bucket_calendar.py # Calendar/bucket tests
│   │   └── test_*.py
│   ├── gui/
│   │   ├── __init__.py
│   │   └── test_widgets.py        # Dashboard widget tests
│   ├── test_cli.py                # CLI command tests
│   └── test_*.py
├── data/                          # Data directory (created at runtime)
│   ├── raw/
│   │   ├── prices/                # OHLCV parquet
│   │   ├── news/                  # News parquet with sentiment
│   │   ├── macro/                 # Macro indicators parquet
│   │   └── onchain/               # On-chain metrics parquet
│   ├── features/                  # Assembled datasets
│   │   └── {freq}_{horizon}/
│   │       ├── dataset.parquet    # Feature matrix + labels
│   │       └── meta.json          # Dataset metadata
│   ├── predictions/               # Batch predictions
│   │   └── {freq}_{horizon}.parquet
│   ├── models/                    # Trained models
│   │   └── {freq}_{horizon}/
│   │       └── xgb.json           # XGBoost booster
│   ├── metrics/                   # Performance metrics
│   │   └── cv_summary.json        # Walk-forward CV results
│   └── autonomous.db              # SQLite state for autonomous agent
├── streamlit/                     # Streamlit dashboard (if present)
│   └── app.py                     # Dashboard entry point
├── docker-compose.yml             # 4 services: api, ingest, monitor, ui
├── Dockerfile                     # Container image definition
├── pyproject.toml                 # Poetry dependencies and metadata
├── poetry.lock                    # Locked dependency versions
├── Makefile                       # Command shortcuts
├── CLAUDE.md                      # This file (Claude instructions)
└── .gitignore                     # Git exclusions (includes .claude, CLAUDE.md)
```

## Directory Purposes

**src/bitbat/**
- Purpose: Main package containing all production code
- Contains: Modules for ingestion, feature engineering, modeling, API, monitoring
- Key files: `cli.py` (entry point), `contracts.py` (validation), `config/default.yaml` (configuration)

**src/bitbat/config/**
- Purpose: Configuration management
- Contains: YAML loader, default settings, runtime config access
- Key files: `default.yaml` (all tunable parameters)

**src/bitbat/ingest/**
- Purpose: External data fetchers
- Contains: Price, news, macro, on-chain data sources
- Key files: `prices.py` (yfinance), `news_gdelt.py`, `news_cryptocompare.py`, `macro_fred.py`, `onchain.py`

**src/bitbat/timealign/**
- Purpose: Time series alignment and leakage prevention
- Contains: UTC normalization, embargo bars, walk-forward logic
- Key files: `calendar.py`, `bucket.py`, `purging.py`

**src/bitbat/features/**
- Purpose: Feature engineering transformations
- Contains: Price indicators, sentiment, volatility, macro, on-chain features
- Key files: `price.py` (ATR, MACD, RSI, OBV), `sentiment.py`, `volatility.py` (GARCH), `macro.py`, `onchain.py`

**src/bitbat/labeling/**
- Purpose: Label generation from price data
- Contains: Forward return computation, 3-class classification
- Key files: `returns.py`

**src/bitbat/dataset/**
- Purpose: Dataset assembly and validation
- Contains: Feature merging, label alignment, walk-forward splitting
- Key files: `build.py` (build_xy entrypoint), `splits.py`, `meta.py`

**src/bitbat/model/**
- Purpose: Model training, inference, evaluation
- Contains: XGBoost training, persistence, walk-forward CV, ensemble methods
- Key files: `train.py` (fit_xgb), `infer.py`, `evaluate.py`, `walk_forward.py`, `persist.py`

**src/bitbat/backtest/**
- Purpose: Strategy backtesting
- Contains: Trade execution, performance metrics
- Key files: `engine.py` (run strategy), `metrics.py` (Sharpe, drawdown)

**src/bitbat/analytics/**
- Purpose: Analysis, explainability, reporting
- Contains: Feature importance, SHAP, Monte Carlo, backtest reports
- Key files: `feature_analysis.py`, `explainer.py`, `monte_carlo.py`, `backtest_report.py`

**src/bitbat/io/**
- Purpose: I/O utilities
- Contains: File system and DuckDB helpers
- Key files: `fs.py`, `duck.py`

**src/bitbat/api/**
- Purpose: REST API endpoints
- Contains: FastAPI app, route handlers, request/response schemas
- Key files: `app.py` (factory), `schemas.py`, `routes/` (health, predictions, analytics, metrics)

**src/bitbat/autonomous/**
- Purpose: Autonomous monitoring and retraining
- Contains: Agent loop, drift detection, periodic ingestion, state tracking
- Key files: `agent.py` (main), `orchestrator.py`, `drift.py`, `retrainer.py`, `models.py` (SQLAlchemy), `db.py`

**src/bitbat/gui/**
- Purpose: Streamlit dashboard components
- Contains: Reusable UI widgets
- Key files: `widgets.py`

**tests/**
- Purpose: Test suite organized by module
- Contains: 54 tests covering contracts, features, dataset, model, backtest, autonomous, timealign, CLI, GUI
- Key files: `conftest.py` (fixtures), `features/test_leakage.py` (critical leakage detection)

**data/**
- Purpose: Runtime data directory
- Contains: Raw data, features, models, predictions, metrics, autonomous state
- Key files: `raw/`, `features/`, `models/`, `autonomous.db`

## Key File Locations

**Entry Points:**
- `src/bitbat/cli.py`: CLI orchestrator — invoked by `poetry run bitbat`
- `src/bitbat/api/app.py`: FastAPI app — invoked by `poetry run uvicorn bitbat.api.app:app`
- `src/bitbat/autonomous/agent.py`: Autonomous agent — invoked by `bitbat monitor start`
- `streamlit/app.py`: Dashboard (outside src/)

**Configuration:**
- `src/bitbat/config/default.yaml`: Single source of truth for all tunable parameters
- `src/bitbat/config/loader.py`: Config loading and runtime access

**Core Logic:**
- `src/bitbat/contracts.py`: Schema validation (ensures all parquets meet contracts)
- `src/bitbat/dataset/build.py`: Dataset assembly (build_xy entrypoint)
- `src/bitbat/model/train.py`: Model training (fit_xgb entrypoint)
- `src/bitbat/model/walk_forward.py`: Walk-forward CV orchestration
- `src/bitbat/backtest/engine.py`: Strategy backtesting (run entrypoint)
- `src/bitbat/autonomous/orchestrator.py`: Task scheduling and coordination

**Testing:**
- `tests/contracts/test_contracts.py`: Schema validation tests
- `tests/features/test_leakage.py`: Future data leakage detection (critical)
- `tests/dataset/test_build_xy.py`: Dataset assembly correctness
- `tests/model/test_train.py`: Model training tests
- `tests/model/test_walk_forward.py`: Walk-forward CV tests
- `tests/conftest.py`: Pytest fixtures and configuration

## Naming Conventions

**Files:**
- Module files: `snake_case.py` (e.g., `price_ingestion.py`, `walk_forward.py`)
- Test files: `test_*.py` (e.g., `test_contracts.py`, `test_build_xy.py`)
- Config files: `*.yaml` or `default.yaml`
- Data files: `*.parquet`, `*.json`, `*.db`

**Directories:**
- Package directories: `snake_case/` (e.g., `src/bitbat/`, `src/bitbat/ingest/`)
- Test directories: `tests/` with subdirectories matching `src/bitbat/` structure
- Data directories: `data/raw/`, `data/features/`, `data/models/`, etc.

**Python Naming:**
- Classes: `PascalCase` (e.g., `ContractError`, `DatasetMeta`)
- Functions: `snake_case` (e.g., `ensure_prices_contract()`, `build_xy()`, `fit_xgb()`)
- Constants: `UPPER_SNAKE_CASE` (e.g., config keys in YAML)
- Private functions: `_snake_case` prefix (e.g., `_ensure_datetime()`, `_generate_price_features()`)

## Where to Add New Code

**New Feature (price indicator, sentiment metric, macro variable):**
- Implementation: `src/bitbat/features/{category}.py` (price, sentiment, macro, onchain, volatility)
- Tests: `tests/features/test_{category}.py`
- Integration: Add feature call to `src/bitbat/dataset/build.py:_generate_price_features()` with config toggle if optional
- Config: Add toggle to `src/bitbat/config/default.yaml` (e.g., `enable_{feature}`)

**New Data Source (external API, blockchain endpoint):**
- Implementation: `src/bitbat/ingest/{source_name}.py` (e.g., `ingest/prices.py`, `ingest/news_gdelt.py`)
- Tests: `tests/ingest/` (if not present, create directory)
- CLI Integration: Add command to `src/bitbat/cli.py` under appropriate command group
- Autonomous Integration: Create `src/bitbat/autonomous/{source_name}_ingestion.py` if periodic fetch needed
- Config: Add connection params and toggles to `src/bitbat/config/default.yaml`

**New API Endpoint:**
- Route handler: Create new file in `src/bitbat/api/routes/` (e.g., `routes/custom.py`)
- Schemas: Add request/response models to `src/bitbat/api/schemas.py`
- Registration: Import and register router in `src/bitbat/api/app.py:create_app()`
- Tests: Create `tests/api/test_{endpoint}.py` if test directory doesn't exist

**New Model Type (replace XGBoost, add alternative):**
- Implementation: Create `src/bitbat/model/{model_name}.py` (e.g., `model/lgb.py` for LightGBM)
- Interface: Implement functions matching `train.py`, `infer.py`, `persist.py` signatures
- Integration: Update `src/bitbat/model/train.py` conditional logic or create config toggle
- Tests: Add `tests/model/test_{model_name}.py`
- Config: Add model selection to `src/bitbat/config/default.yaml`

**New Autonomous Task (custom monitoring, custom retraining logic):**
- Implementation: Create `src/bitbat/autonomous/{task_name}.py`
- Orchestration: Register in `src/bitbat/autonomous/orchestrator.py`
- State Tracking: Add SQLAlchemy model to `src/bitbat/autonomous/models.py` if state needed
- Tests: Add `tests/autonomous/test_{task_name}.py`
- Config: Add timing/threshold params to `src/bitbat/config/default.yaml`

**Utilities:**
- Shared helpers: `src/bitbat/io/{util_name}.py` (file system, database, parsing)
- Time utilities: `src/bitbat/timealign/{util_name}.py` (calendar, bucketing)
- Analytics helpers: `src/bitbat/analytics/{util_name}.py`

## Special Directories

**data/**
- Purpose: Runtime data storage
- Generated: Yes (created at runtime by CLI commands)
- Committed: No (in .gitignore)
- Contents: Raw data (parquet), features, models, predictions, metrics, autonomous state (SQLite)
- Organization: `raw/{source}/`, `features/{freq}_{horizon}/`, `models/{freq}_{horizon}/`, `predictions/`, `metrics/`

**models/**
- Purpose: Trained model artifacts
- Generated: Yes (by `bitbat model train`)
- Committed: No (in .gitignore)
- Structure: `models/{freq}_{horizon}/xgb.json` and feature importance mappings

**metrics/**
- Purpose: Walk-forward CV and performance metrics
- Generated: Yes (by `bitbat model train`, `bitbat backtest run`)
- Committed: No (in .gitignore)
- Key files: `cv_summary.json` (train/test scores by fold), `backtest_summary.json`

**tests/**
- Purpose: Test suite
- Generated: No (committed to repo)
- Committed: Yes
- Structure: Mirror of `src/bitbat/` with `test_*.py` files

---

*Structure analysis: 2026-02-24*
