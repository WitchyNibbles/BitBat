# Testing & Quality

Quality assurance centers on preventing leakage, enforcing contracts, and maintaining reproducibility.

## Lint & Type Checks

- **Ruff** (`poetry run ruff check src tests`): Style, import sorting, code smells.
- **Mypy** (`poetry run mypy src tests`): Ensures static typing across modules and tests.
- **Make target** (`make lint`): Runs the same Ruff + Mypy checks from the repository Makefile.

CI should run both before executing the test suite.

## Pytest Suites

- `tests/backtest/`: Strategy engine and backtest metrics output checks.
- `tests/contracts/`: Confirms contract validators reject/accept appropriate schemas.
- `tests/ingest/`: Checks idempotent ingestion and contract compliance for prices/news.
- `tests/features/`: Validates price/sentiment feature generators do not leak future data.
- `tests/dataset/`: Asserts dataset assembly structure, metadata capture, and embargoed splits.
- `tests/io/`: Validates parquet IO helpers and DuckDB query helper behavior.
- `tests/labeling/`: Verifies forward-return alignment and threshold classification behavior.
- `tests/model/`: Covers training, inference, persistence, and evaluation guardrails (PR-AUC leakage warning).
- `tests/test_cli.py`: End-to-end scenarios, validating CLI commands stay single-purpose and idempotent.
- `tests/test_config_loader.py`: Covers default config load behavior and runtime config initialization.
- `tests/timealign/`: Guarantees calendar/bucketing/purging uphold temporal ordering.

Run `poetry run pytest` regularly—current suite collects 54 tests (`poetry run pytest --collect-only -q`).

## Guardrails

- **Contracts**: `ensure_prices_contract` and `ensure_news_contract` run in ingestion before persistence; `ensure_feature_contract` runs in dataset build and feature-consuming CLI flows; `ensure_predictions_contract` runs across prediction-producing/consuming CLI flows.
- **Time alignment**: Sentiment aggregation raises if future news is included and only aggregates publications up to each bar timestamp.
- **Embargo**: Walk-forward splits enforce gaps between train/test windows to avoid leakage.
- **PR-AUC warning**: Evaluation prints `[metrics-warning]` if PR-AUC jumps >10 points while class balance stays stable.
- **Idempotency**: Ingestion and batch prediction flows deduplicate on natural keys to avoid duplicates.
- **Metadata**: `meta.json` captures dataset columns, time bounds, class counts, and key run settings (`freq`, `horizon`, `tau`, `seed`, `version`).

## Reproducibility Tips

- Pin seeds in configs; `build_xy` persists metadata including `columns`, `freq`, `horizon`, `tau`, `start`, `end`, `rows`, class counts, `seed`, and `version`.
- Store environment details (Python, dependency versions) when shipping models.
- Archive `dataset.parquet`, `models`, and `metrics` artifacts alongside experiment logs.

## Pre-commit Hooks

Configured in `.pre-commit-config.yaml`:

- `ruff` with `--fix`
- `ruff-format`
- `black`
- `mypy` (with `types-requests` additional dependency)

Run locally (after installing `pre-commit`) with:

- `pre-commit run --all-files`

For contract specifications, see [Data Contracts](./data-contracts.md).
