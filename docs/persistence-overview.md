## BitBat Persistence Overview

This document explains **what BitBat stores**, **where it stores it** on a local machine, and **which folders you should back up** to preserve your models and history.

The design assumes a **single-user local desktop** install with everything rooted under the project directory.

---

## High-level storage layout

- **Database (`data/autonomous.db`)**
  - SQLite database managed by `AutonomousDB`.
  - Stores:
    - `prediction_outcomes`: every live prediction, its realized return, direction, and correctness.
    - `model_versions`: metadata for each trained model (freq/horizon, training window, sample count, CV score, feature list, deployment flags).
    - `performance_snapshots`: rolling hit-rate, Sharpe, drawdown, calibration metrics.
    - `retraining_events`: who/what triggered retraining, status, and timing.
    - `system_logs`: internal log messages from the autonomous pipeline.

- **Raw data (`data/raw/*`)**
  - **Prices**: `data/raw/prices/…`
    - Historical backfill from Yahoo: `btcusd_yf_<freq>.parquet` (for example `btcusd_yf_1h.parquet`).
    - Incremental ingestion from the monitoring agent under date-partitioned subdirectories.
  - **News**: `data/raw/news/<source>_1h/*.parquet`
    - Currently `cryptocompare_1h` or `gdelt_1h` depending on config.
  - **Macro**: `data/raw/macro/fred.parquet`
  - **On-chain**: `data/raw/onchain/blockchain_info.parquet`

- **Feature datasets (`data/features/{freq}_{horizon}/*`)**
  - `dataset.parquet`: model-ready feature matrix and labels built by `build_xy` or the Pipeline page.
  - `meta.json`: `DatasetMeta` describing label horizon, tau, and feature configuration.

- **Batch prediction files (`data/predictions/{freq}_{horizon}.parquet`)**
  - Used by the advanced Pipeline + backtest tools.
  - Contains probabilities (`p_up`, `p_down`), realized labels and returns, and model metadata for batch runs.

- **Models (`models/{freq}_{horizon}/xgb.json`)**
  - XGBoost model artifacts trained by:
    - one-click training (`one_click_train` orchestrator).
    - manual training from the Pipeline page.
    - automated retraining (`AutoRetrainer`).

- **Metrics and monitoring artifacts (`metrics/*`)**
  - `classification_metrics.json`, `confusion_matrix.png`: produced by `bitbat.model.evaluate.classification_metrics`.
  - `cv_summary.json`: cross-validation summary used by `AutoRetrainer`.
  - `live_*.json`: live monitoring exports from the Pipeline Monitor page.

- **Logs and heartbeats**
  - `logs/monitoring_agent.log`: long-running monitoring agent log file.
  - `data/monitoring_agent_heartbeat.json` (or near `autonomous.db`): liveness + configuration summary for the CLI monitoring agent.

---

## Configuration and defaults

Most components read paths from the YAML config files via `bitbat.config.loader`:

- **Data directory (`data_dir`)**
  - Default: `"data"`.
  - Used by:
    - Ingestion services (`PriceIngestionService`, `MacroIngestionService`, `OnchainIngestionService`).
    - Feature builders and advanced Pipeline pages.
    - Autonomous predictor and validator when reading raw prices/news/macro/on-chain data.

- **Autonomous database URL (`autonomous.database_url`)**
  - Default: `sqlite:///data/autonomous.db`.
  - Used by:
    - `AutonomousDB` (directly and via `create_database_engine`).
    - `MonitoringAgent` (via CLI script and Quick Start).
    - CLI commands that register models or store predictions.

- **Models directory**
  - Implicitly `models/{freq}_{horizon}/xgb.json` inside `fit_xgb` and `LivePredictor`.
  - No extra configuration is required for local desktop usage.

- **Metrics directory**
  - Always a relative `metrics/` folder in the project root.
  - Created automatically when evaluation code runs.

On a typical local setup, you don’t need to override any of these; keeping the project folder intact is enough to preserve your data.

---

## What to back up (or move to another machine)

To migrate BitBat with all its history and models, you should back up:

- `data/`
  - `data/autonomous.db`
  - `data/raw/**`
  - `data/features/**`
  - `data/predictions/**`
  - any JSON heartbeat and live-metrics files under `data/`.
- `models/`
  - all subdirectories such as `models/1h_4h/xgb.json`.
- `metrics/` (optional but recommended)
  - keeps historical evaluation and confusion matrices used by the UI.
- `config/`
  - especially `config/user_config.yaml` if you have customized settings or autonomous thresholds.

Copying these folders into a fresh clone on another machine should restore:

- full prediction history and realized outcomes in the dashboard
- performance and history charts
- all trained model artifacts
- the ability to continue training and monitoring from where you left off.

---

## How the autonomous loop uses persistence

When the monitoring loop is running (either from Quick Start or via `scripts/run_monitoring_agent.py`):

- it **reads** the latest raw prices/news/macro/on-chain data from `data/raw/**` using the configured `data_dir`.
- it **writes** new predictions, realized outcomes, and performance snapshots into `data/autonomous.db`.
- it **logs** drift decisions and retraining events into `data/autonomous.db` and `logs/monitoring_agent.log`.
- when drift is detected, `AutoRetrainer`:
  - rebuilds features under `data/features/{freq}_{horizon}/`,
  - trains and saves a new model under `models/{freq}_{horizon}/`,
  - records a new `model_versions` row and optionally marks it active.

Because all of these operations are file- and DB-based, **restarting Streamlit or the monitoring process does not lose history** as long as `data/` and `models/` remain on disk.

---

## Restoring and verifying after a restart

After restarting your machine or app:

1. Start Streamlit: `streamlit run streamlit/app.py`.
2. Go to **Quick Start**:
   - It will detect any existing model under `models/{freq}_{horizon}` and mark the system as running if you choose to start monitoring.
3. Visit **Performance** and **History** pages:
   - They read directly from `data/autonomous.db` and should show the same predictions, accuracy, and retraining history as before the restart.

If these pages show “no data” after a restart, check that:

- the `data/` directory (especially `data/autonomous.db`) is present,
- your configuration still points to `sqlite:///data/autonomous.db`,
- your working directory when launching Streamlit is the project root.

