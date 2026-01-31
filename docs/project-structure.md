# Project Structure

The repository is organized into the following key directories and modules:

## Top-Level Layout

```
├── data/                   # Data artifacts generated locally during runs
├── docs/                   # Project documentation (this directory)
├── metrics/                # Metrics artifacts (confusion matrices, CV summaries, etc.)
├── models/                 # Persisted model checkpoints per (freq, horizon)
├── predictions/            # Contract-compliant prediction parquet files
├── src/                    # Source code packages
├── tests/                  # Pytest suites covering contracts, ingestion, features, models, CLI, etc.
├── pyproject.toml          # Poetry configuration (deps, linting, typing, pytest)
└── poetry.lock             # Locked dependency versions
```

## Source Packages (`src/bitbat`)

- `bitbat/cli.py`
  - Click-based CLI exposing command groups: `prices`, `news`, `features`, `model`, `backtest`, `batch`, `monitor`.
  - Normalizes configuration, loads contract-compliant datasets, and orchestrates operations.

- `bitbat/contracts.py`
  - Schema validators for prices, news, features, and predictions.
  - Used across ingestion, dataset builds, and prediction flows.

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

- `bitbat/io/`: File-system utilities for reading/writing parquet with consistent settings.

- `bitbat/timealign/`: Calendar/bucketing/purging helpers enforcing temporal alignment.

- `bitbat/labeling/`: Forward return computation and threshold-based classification.

## Tests

- `tests/contracts/`: Contract validator coverage.
- `tests/ingest/`: Ingestion idempotency & schema verification.
- `tests/dataset/`: Dataset build shape/metadata tests, embargo checks.
- `tests/features/`: Feature leakage tests ensuring no future contamination.
- `tests/model/`: Training, inference, evaluation guardrails (PR-AUC warning).
- `tests/test_cli.py`: End-to-end CLI flows with monkeypatched dependencies.
- `tests/timealign/`: Purging and bucketing utilities.

Use this structure as a map when exploring the codebase or extending functionality. Consult the [Usage Guide](./usage-guide.md) for operational commands, and [Data Contracts](./data-contracts.md) for precise schema expectations.
