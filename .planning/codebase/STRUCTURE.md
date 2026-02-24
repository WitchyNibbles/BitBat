# Structure

**Analysis Date:** 2026-02-24

## Repository Layout (Top Level)

- `src/bitbat/` - core application package.
- `tests/` - pytest suite mirroring runtime modules.
- `streamlit/` - dashboard app and page modules.
- `scripts/` - service entrypoint scripts.
- `deployment/` - docker/systemd/nginx/prometheus deployment files.
- `config/` - user-facing runtime overrides (for UI/settings flows).
- `data/`, `models/`, `metrics/` - runtime artifacts.
- `docs/` - project docs and architecture notes.

## Core Package Map (`src/bitbat`)

- `api/` - FastAPI app + routers + schemas.
- `analytics/` - explainability, Monte Carlo, feature and backtest reports.
- `autonomous/` - DB models/repository, monitoring agent, drift/retraining, ingestion services, predictor.
- `backtest/` - strategy engine and performance metrics.
- `config/` - default config YAML + loader.
- `dataset/` - dataset build and split logic.
- `features/` - feature engineering by domain (price/sentiment/macro/onchain/volatility).
- `gui/` - reusable dashboard-side helpers/widgets/presets/timeline utilities.
- `ingest/` - batch data ingestion modules.
- `io/` - parquet and DuckDB helpers.
- `labeling/` - return/target logic.
- `model/` - training, inference, optimization, persistence, walk-forward.
- `timealign/` - UTC/calendar/purging/bucket logic.
- `cli.py` - Click command entrypoint.
- `contracts.py` - cross-pipeline schema validators.

## UI Structure (`streamlit/`)

- `streamlit/app.py` - main dashboard home.
- `streamlit/pages/` - multi-page UX:
  - quick start, settings, performance, system, alerts, analytics, history, backtest, pipeline, about.
- `streamlit/style.py` - shared visual styling helper.
- `streamlit/app_pipeline_backup.py` - legacy/backup UI implementation retained in repo.

## Service Entrypoints (`scripts/`)

- `scripts/run_ingestion_service.py` - continuous price+news ingestion loop.
- `scripts/run_monitoring_agent.py` - autonomous monitoring process with heartbeat.
- `scripts/init_autonomous_db.py` - schema initialization/reset utility.

## Test Structure (`tests/`)

- Areas align with runtime modules:
  - `tests/api/`, `tests/autonomous/`, `tests/model/`, `tests/ingest/`, `tests/features/`, `tests/dataset/`, `tests/backtest/`, `tests/analytics/`, `tests/gui/`, `tests/io/`, etc.
- No global `conftest.py` currently present; fixtures are mostly module-local.
- Includes milestone/session coverage tests (`test_phase*_complete.py`, `test_session*_complete.py`).

## Deployment Structure (`deployment/`)

- `deployment/nginx.conf` - reverse-proxy routing.
- `deployment/prometheus.yml` - scrape config.
- `deployment/docker-compose.monitoring.yml` - monitoring stack overlay.
- `deployment/bitbat-ingest.service` and `deployment/bitbat-monitor.service` - systemd units.

## Artifact and State Directories

- `data/raw/` - ingested market/news/macro/on-chain data.
- `data/features/` - assembled model datasets by `freq_horizon`.
- `data/predictions/` - prediction parquet outputs.
- `data/autonomous.db` - operational state database.
- `models/{freq}_{horizon}/xgb.json` - trained model artifact.
- `metrics/` - CV summaries, live metrics JSON, plots.

## Naming and Organization Patterns

- Module names are snake_case; package segmentation reflects pipeline stages.
- Test files use `test_*.py` naming.
- Many paths encode runtime dimensions as `{freq}_{horizon}`.
- Default config and runtime paths are centralized in `src/bitbat/config/default.yaml`.

