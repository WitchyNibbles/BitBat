# Testing & Quality

Quality assurance centers on preventing leakage, enforcing contracts, and maintaining reproducibility.

## Lint & Type Checks

- **Ruff** (`poetry run ruff check src tests`): Style, import sorting, code smells.
- **Mypy** (`poetry run mypy src tests`): Ensures static typing across modules and tests.

CI should run both before executing the test suite.

## Pytest Suites

- `tests/contracts/`: Confirms contract validators reject/accept appropriate schemas.
- `tests/ingest/`: Checks idempotent ingestion and contract compliance for prices/news.
- `tests/features/`: Validates price/sentiment feature generators do not leak future data.
- `tests/dataset/`: Asserts dataset assembly structure, metadata capture, and embargoed splits.
- `tests/model/`: Covers training, inference, persistence, and evaluation guardrails (PR-AUC leakage warning).
- `tests/test_cli.py`: End-to-end scenarios, validating CLI commands stay single-purpose and idempotent.
- `tests/timealign/`: Guarantees calendar/bucketing/purging uphold temporal ordering.

Run `poetry run pytest` regularly—current suite includes 50+ tests.

## Guardrails

- **Contracts**: `ensure_prices_contract`, `ensure_news_contract`, `ensure_feature_contract`, `ensure_predictions_contract` run before writing critical datasets.
- **Time alignment**: Sentiment aggregation raises if future timestamps slip in; dataset indices filtered to `<=` bar time.
- **Embargo**: Walk-forward splits enforce gaps between train/test windows to avoid leakage.
- **PR-AUC warning**: Evaluation prints `[metrics-warning]` if PR-AUC jumps >10 points while class balance stays stable.
- **Idempotency**: Ingestion/backfill operations deduplicate on natural keys to avoid duplicates.
- **Metadata**: `meta.json` records freq, horizon, tau, seed, version for reproducibility.

## Reproducibility Tips

- Pin seeds in configs; `build_xy` persists the seed/version in metadata.
- Store environment details (Python, dependency versions) when shipping models.
- Archive `dataset.parquet`, `models`, and `metrics` artifacts alongside experiment logs.

For contract specifications, see [Data Contracts](./data-contracts.md).
