# Project Structure

The repository is organized into the following key directories and modules:

## Top-Level Layout

```
├── data/                   # Data artifacts generated locally during runs
├── docs/                   # Project documentation (this directory)
├── metrics/                # Metrics artifacts (confusion matrices, CV summaries, etc.)
├── models/                 # Persisted model checkpoints per (freq, horizon)
├── predictions/            # Legacy sample predictions (runtime uses data/predictions/)
├── src/                    # Source code packages
├── streamlit/              # Streamlit web UI
├── tests/                  # Pytest suites covering contracts, ingestion, features, models, CLI, etc.
├── pyproject.toml          # Poetry configuration (deps, linting, typing, pytest)
└── poetry.lock             # Locked dependency versions
```

Runtime output locations are configuration-driven via `data_dir` (default `data/`):

- `${data_dir}/raw/` for ingested prices/news
- `${data_dir}/features/{freq}_{horizon}/` for `dataset.parquet` and `meta.json`
- `${data_dir}/predictions/{freq}_{horizon}.parquet` for live predictions
- `models/{freq}_{horizon}/xgb.json` for model artifacts
- `metrics/` for evaluation and monitoring snapshots

## Source Packages (`src/bitbat`)

- `bitbat/cli.py`
  - Click-based CLI exposing command groups: `prices`, `news`, `features`, `model`, `backtest`, `batch`, `monitor`.
  - Normalizes configuration, loads contract-compliant datasets, and orchestrates operations.

- `bitbat/contracts.py`
  - Schema validators for prices, news, features, and predictions.
  - Used across ingestion, dataset builds, and prediction flows.

- `bitbat/config/`
  - `loader.py`: Runtime/default config resolution (`BITBAT_CONFIG` support) and cached config accessors.

- `bitbat/ingest/`
  - `prices.py`: yfinance ingestion with partitioned parquet output, applying price contract.
  - `news_gdelt.py`: GDELT ingestion, deduping by URL and enforcing news contract.

- `bitbat/features/`
  - `price.py`: Lagged returns, volatility, ATR, MACD, RSI, OBV features.
  - `sentiment.py`: VADER-based scoring and aggregation with future-leak prevention.

- `bitbat/dataset/`
  - `build.py`: Constructs feature+label datasets, persists `dataset.parquet`, records metadata (freq/horizon/tau/seed/version).
  - `splits.py`: Walk-forward splitter with embargo bars.

- `bitbat/model/`
  - `train.py`: XGBoost training with class weighting, persists models per freq/horizon.
  - `infer.py`: Single-bar inference helper.
  - `evaluate.py`: Classification metrics with leakage warning guardrail.
  - `persist.py`: Model save/load wrappers.

- `bitbat/backtest/`
  - `engine.py`: Simple strategy simulator turning probabilities into positions, factoring costs.
  - `metrics.py`: Summaries & plots (Sharpe, drawdown, hit rate).

- `bitbat/io/`
  - `fs.py`: File-system parquet helpers (`read_parquet`, `write_parquet`).
  - `duck.py`: DuckDB SQL helper for in-memory query execution.

- `bitbat/timealign/`: Calendar/bucketing/purging helpers enforcing temporal alignment.

- `bitbat/labeling/`: Forward return computation and threshold-based classification.

## Tests

- `tests/backtest/`: Strategy engine and backtest metrics coverage.
- `tests/contracts/`: Contract validator coverage.
- `tests/ingest/`: Ingestion idempotency & schema verification.
- `tests/dataset/`: Dataset build shape/metadata tests, embargo checks.
- `tests/features/`: Feature leakage tests ensuring no future contamination.
- `tests/io/`: Parquet and IO helper behavior.
- `tests/labeling/`: Forward returns and threshold labeling behavior.
- `tests/model/`: Training, inference, evaluation guardrails (PR-AUC warning).
- `tests/test_cli.py`: End-to-end CLI flows with monkeypatched dependencies.
- `tests/test_config_loader.py`: Config loading, env override, and runtime cache behavior.
- `tests/timealign/`: Purging and bucketing utilities.

Use this structure as a map when exploring the codebase or extending functionality. Consult the [Usage Guide](./usage-guide.md) for operational commands, and [Data Contracts](./data-contracts.md) for precise schema expectations.
