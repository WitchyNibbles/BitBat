# Technology Stack

**Analysis Date:** 2026-02-24

## Languages

**Primary:**
- Python 3.11+ - End-to-end ML pipeline, data processing, model training, API, CLI, monitoring agent

## Runtime

**Environment:**
- CPython 3.11+ (specified in `pyproject.toml`)

**Package Manager:**
- Poetry 1.x (dependency management)
- Lockfile: `poetry.lock` (present)

## Frameworks

**Core ML/Data:**
- pandas 2.2.2 - DataFrames, time series manipulation
- XGBoost 2.1.0 - Gradient boosted tree models for price direction prediction
- scikit-learn 1.4.0 - Feature preprocessing, metrics, model evaluation
- optuna 4.7.0 - Hyperparameter optimization via Bayesian search

**Data Storage & Processing:**
- DuckDB 1.1.2 - In-process SQL queries on Parquet files (`bitbat/io/`)
- PyArrow 17.0.0 - Parquet format serialization
- SQLAlchemy 2.0.37 - ORM for SQLite autonomous agent database

**API/Web:**
- FastAPI 0.129.0 - REST API server with OpenAPI/Swagger docs (`bitbat/api/`)
- uvicorn 0.40.0 - ASGI application server (with standard extras: HTTP/2, TLS)
- httpx 0.28.1 - Async HTTP client for internal requests

**Dashboard/UI:**
- Streamlit 1.38.0 - Interactive web dashboard (`streamlit/app.py`)

**CLI:**
- Click (via poetry scripts) - Command-line interface (`bitbat/cli.py`)

**Time Series & Volatility:**
- arch 7.0 - GARCH models for volatility estimation (`bitbat/features/volatility.py`)

**Testing:**
- pytest 8.2.0 - Test runner and framework

**Visualization:**
- matplotlib 3.10.7 - Static plots
- plotly 6.5.2 - Interactive charts

## Key Dependencies

**Critical Data Sources:**
- yfinance 0.2.40 - Yahoo Finance API client for OHLCV price data (`bitbat/ingest/prices.py`)
- requests 2.32.3 - HTTP client for GDELT, CryptoCompare, FRED, blockchain.info APIs

**Sentiment Analysis:**
- vaderSentiment 3.3.2 - VADER lexicon-based sentiment scoring on news text (`bitbat/features/`)

**Configuration:**
- PyYAML 6.0.0 - YAML config parsing (`bitbat/config/`)

## Configuration

**Environment:**
- Config driven by `src/bitbat/config/default.yaml`
- Pipeline parameters: `freq`, `horizon`, `tau` (label threshold), `seed`
- Feature toggles: `enable_sentiment`, `enable_garch`, `enable_macro`, `enable_onchain`
- Autonomous system config: database URL, monitoring intervals, drift thresholds, alert channels
- Data API keys: `fred_api_key` for FRED Federal Reserve data

**Build:**
- `pyproject.toml` - Poetry configuration, all dependencies, build backend
- `Makefile` - Shortcuts for test, lint, format commands
- `.pytest.ini` (embedded in pyproject.toml) - pytest settings, test discovery

**Development:**
- Black 24.4.0 - Code formatter (100 char line length)
- Ruff 0.5.0 - Fast linter (E, F, B, I, UP, S, C4, RET, SIM rules)
- mypy 1.11.0 - Static type checker (strict mode enabled, PEP 484)

## Platform Requirements

**Development:**
- Python 3.11+
- Poetry for dependency/virtual environment management
- Make (for shortcuts)
- Git

**Production:**
- Python 3.11+ runtime
- FastAPI/uvicorn for API server (port 8000 by default)
- Streamlit for dashboard (port 8501 by default)
- Docker & Docker Compose (optional, 4 services in `docker-compose.yml`)
- SQLite for autonomous agent state persistence (`data/autonomous.db`)

---

*Stack analysis: 2026-02-24*
