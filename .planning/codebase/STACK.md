# Technology Stack

**Analysis Date:** 2026-02-24

## Runtime and Languages

- Primary language: Python (project targets Python 3.11+ in `pyproject.toml`, CI uses 3.12).
- Packaging/build: Poetry (`pyproject.toml`, `poetry.lock`, `[tool.poetry.scripts] bitbat = "src.bitbat.cli:main"`).
- Process model: CLI + API + Streamlit UI + long-running ingestion/monitoring workers.

## Core Libraries

### Data and ML

- `pandas`, `numpy` for time-series and feature engineering across `src/bitbat/features/` and `src/bitbat/dataset/`.
- `xgboost` for training/inference in `src/bitbat/model/train.py`, `src/bitbat/model/infer.py`.
- `scikit-learn` and `optuna` for evaluation/optimization paths in `src/bitbat/model/optimize.py` and related model tests.
- `arch` (optional) for GARCH features in `src/bitbat/features/volatility.py`.

### Storage and Persistence

- Parquet via `pyarrow` through helpers in `src/bitbat/io/fs.py`.
- `duckdb` for in-memory SQL querying in `src/bitbat/io/duck.py`.
- `sqlalchemy` with SQLite backend for autonomous state in `src/bitbat/autonomous/models.py` and `src/bitbat/autonomous/db.py`.

### APIs and UI

- FastAPI app factory in `src/bitbat/api/app.py`.
- Uvicorn ASGI server for API runtime (`uvicorn bitbat.api.app:app`).
- Streamlit multi-page app in `streamlit/app.py` and `streamlit/pages/`.

### External Data and HTTP

- `yfinance` for BTC OHLCV ingestion (`src/bitbat/ingest/prices.py`, `src/bitbat/autonomous/price_ingestion.py`).
- `requests` for GDELT/CryptoCompare/FRED/blockchain.info and webhook alerting.
- Optional `praw` integration used dynamically in `src/bitbat/autonomous/news_ingestion.py`.
- `vaderSentiment` for headline sentiment scoring (`src/bitbat/features/sentiment.py` and ingestion modules).

## Internal Application Surfaces

- CLI orchestration: `src/bitbat/cli.py`.
- REST API routes: `src/bitbat/api/routes/health.py`, `predictions.py`, `analytics.py`, `metrics.py`.
- Autonomous monitoring/retraining loop: `src/bitbat/autonomous/agent.py`, `continuous_trainer.py`, `retrainer.py`.
- Backtesting and analytics: `src/bitbat/backtest/`, `src/bitbat/analytics/`.

## Configuration System

- Default runtime config: `src/bitbat/config/default.yaml`.
- Loader utilities: `src/bitbat/config/loader.py`.
- Optional override via `BITBAT_CONFIG`.
- UI-level user config: `config/user_config.yaml`.

## Local Artifact Layout

- Raw and processed data: `data/`.
- Trained models: `models/`.
- Monitoring/evaluation outputs: `metrics/`.
- Prediction parquet outputs: `data/predictions/`.
- Autonomous database: `data/autonomous.db`.

## Developer Tooling

- Formatters/linters/type checks: Ruff, Black, MyPy configured in `pyproject.toml`.
- Test runner: Pytest with config in `pyproject.toml` (`[tool.pytest.ini_options]`).
- Convenience commands: `Makefile` (`fmt`, `lint`, `test`, `streamlit`).
- CI pipeline: `.github/workflows/ci.yml` (lint, test, docker build).
- Pre-commit hooks: `.pre-commit-config.yaml`.

## Deployment Stack

- Docker build: `Dockerfile` (multi-stage poetry -> runtime image).
- Compose services: `docker-compose.yml` (API, ingest, monitor, UI, optional nginx profile).
- Optional monitoring overlay: `deployment/docker-compose.monitoring.yml` (Prometheus + Grafana).
- Optional systemd units: `deployment/bitbat-ingest.service`, `deployment/bitbat-monitor.service`.

