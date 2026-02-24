# Architecture

**Analysis Date:** 2026-02-24

## High-Level Style

- Predominantly layered, module-oriented Python architecture.
- Core flow is data pipeline oriented:
  - ingest -> feature build -> dataset contract -> train -> infer -> evaluate/monitor.
- Runtime is split across four entrypoint families:
  - CLI (`src/bitbat/cli.py`)
  - API (`src/bitbat/api/app.py`)
  - Streamlit UI (`streamlit/app.py`)
  - Long-running services (`scripts/run_ingestion_service.py`, `scripts/run_monitoring_agent.py`)

## Core Architectural Layers

### 1) Ingestion Layer

- Batch ingestion modules in `src/bitbat/ingest/` fetch prices/news/macro/on-chain inputs.
- Continuous ingestion modules in `src/bitbat/autonomous/` fetch rolling updates.
- All ingest outputs normalized to contracts before persistence.

### 2) Feature and Label Layer

- Feature generation modules under `src/bitbat/features/`.
- Label and return logic under `src/bitbat/labeling/`.
- UTC alignment and leakage controls in `src/bitbat/timealign/`.

### 3) Dataset Assembly Layer

- `src/bitbat/dataset/build.py` builds contract-compliant feature datasets.
- `src/bitbat/dataset/splits.py` and model walk-forward code manage CV windows/embargo.

### 4) Model Layer

- Train/persist/infer/evaluate modules under `src/bitbat/model/`.
- XGBoost booster persisted at `models/{freq}_{horizon}/xgb.json`.

### 5) Decision and Monitoring Layer

- Strategy backtests in `src/bitbat/backtest/`.
- Autonomous loop in `src/bitbat/autonomous/agent.py`:
  - validates realized predictions,
  - computes drift metrics,
  - optionally triggers retraining,
  - sends alerts.

### 6) Presentation Layer

- FastAPI routers in `src/bitbat/api/routes/` expose health, predictions, analytics, Prometheus metrics.
- Streamlit pages consume DB/files and user config for dashboard workflows.

## Data Flow (Training)

1. Price/news/macro/on-chain data fetched to `data/raw/...`.
2. `build_xy` merges features and forward returns (`src/bitbat/dataset/build.py`).
3. Contract validation via `src/bitbat/contracts.py`.
4. Dataset persisted to `data/features/{freq}_{horizon}/dataset.parquet` + `meta.json`.
5. Model training in `src/bitbat/model/train.py` saves booster to `models/`.
6. Evaluation writes metrics/artifacts to `metrics/`.

## Data Flow (Live/Autonomous)

1. Ingestion services refresh recent price/news and optional macro/on-chain.
2. Predictor (`src/bitbat/autonomous/predictor.py`) generates latest prediction.
3. Prediction stored in SQLite via `AutonomousDB.store_prediction`.
4. Validator resolves matured predictions and computes correctness.
5. Drift detector evaluates degradation windows.
6. Continuous trainer/retrainer may produce/deploy a new model.

## Persistence Model

- File-based artifacts:
  - parquet datasets, predictions, raw inputs (`data/`)
  - model files (`models/`)
  - metrics (`metrics/`)
- Relational state:
  - SQLite tables defined in `src/bitbat/autonomous/models.py`.
  - Includes `prediction_outcomes`, `model_versions`, `performance_snapshots`, `retraining_events`, `system_logs`.

## Contract and Validation Boundary

- Contract functions in `src/bitbat/contracts.py` form explicit schema/type boundaries.
- API response schemas in `src/bitbat/api/schemas.py` constrain HTTP payloads.
- Many pathways fail fast with typed exceptions (`ContractError`, `ClickException`, HTTP errors).

## Operational Topology

- Local/dev:
  - Poetry-driven commands (`make`, `poetry run bitbat ...`).
- Containerized:
  - API, ingest worker, monitoring worker, Streamlit UI, optional nginx reverse proxy.
- Monitoring stack:
  - Prometheus + Grafana overlay compose file.

## Notable Couplings

- Multiple modules assume fixed default pairs (`1h/4h` or `5m/30m`) depending on subsystem.
- A single SQLite file (`data/autonomous.db`) is shared by API, monitor, and UI readers.
- There are parallel implementations for ingestion (batch vs autonomous) with similar responsibilities.

