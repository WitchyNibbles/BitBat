# Concerns

**Analysis Date:** 2026-02-24

## High Priority

### 1) Retraining command mismatch likely breaks autonomous retraining

- Evidence:
  - `src/bitbat/autonomous/retrainer.py` invokes `bitbat features build --tau ...`.
  - `src/bitbat/cli.py` defines `features build` with only `--start` and `--end` options.
- Impact:
  - Retraining flow can fail at runtime before rebuilding dataset/model.
- Additional mismatch:
  - Retrainer `_read_cv_score()` expects `average_balanced_accuracy`, but CLI CV writes `average_rmse`/`average_mae` to `metrics/cv_summary.json`.

### 2) Multiple defaults for freq/horizon across subsystems

- Evidence:
  - Config default is `freq: 5m`, `horizon: 30m` in `src/bitbat/config/default.yaml`.
  - API health/analytics/predictions routes default to `1h`/`4h`.
  - Ingestion service script hardcodes `1h` price interval (`scripts/run_ingestion_service.py`).
- Impact:
  - Services can read/write different paths than expected, causing missing file/model false negatives.

### 3) Duplicate ingestion implementations increase drift risk

- Evidence:
  - Batch ingestion modules in `src/bitbat/ingest/`.
  - Autonomous ingestion services in `src/bitbat/autonomous/` with overlapping responsibilities.
- Impact:
  - Behavior divergence over time (partition schemes, dedupe logic, error handling) can create inconsistent datasets.

## Medium Priority

### 4) Broad exception swallowing hides operational failures

- Evidence:
  - Frequent `except Exception` patterns across monitor loops, UI pages, widgets, and ingestion/retraining modules.
  - Examples: `src/bitbat/autonomous/agent.py`, `scripts/run_monitoring_agent.py`, `src/bitbat/api/routes/metrics.py`, multiple Streamlit pages.
- Impact:
  - Silent degradation; difficult root-cause diagnosis.

### 5) API has no built-in authentication/authorization

- Evidence:
  - FastAPI app and routers expose health/predictions/analytics/metrics without auth middleware.
  - `deployment/nginx.conf` proxies routes but does not add auth controls by default.
- Impact:
  - Unsafe if exposed beyond trusted network boundaries.

### 6) Single SQLite file as shared operational state

- Evidence:
  - `data/autonomous.db` used by API readers, monitoring writer, UI readers.
- Impact:
  - Concurrency and file-lock sensitivity under heavier multi-process usage.

## Low Priority

### 7) Repository hygiene artifacts in tracked tree

- Evidence:
  - Compiled artifacts under `src/bitbat/__pycache__/...` appear in file listing.
  - Large backup UI file `streamlit/app_pipeline_backup.py` retained alongside active app.
- Impact:
  - Noise in code navigation/review and potential confusion about active code path.

### 8) Runtime secrets are config/env based but guardrails are process-level only

- Evidence:
  - Alerting/API keys live in config/env (`src/bitbat/config/default.yaml`, environment variables used by scripts/services).
- Impact:
  - Requires strict deployment discipline; no central secret manager abstraction in code.

## Monitoring Recommendations

- Add automated checks that fail CI when CLI/retrainer contracts drift.
- Standardize default `freq/horizon` behavior across API, CLI, scripts, and config.
- Consolidate shared ingestion behavior into common utilities to reduce duplication.
- Tighten exception boundaries: fail fast on critical paths, downgrade only non-critical branches.

