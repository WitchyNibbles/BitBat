# Usage Guide

This guide outlines typical workflows using the `bitbat` CLI. All commands are implemented in `src/bitbat/cli.py` and validated by contract-enforcing helpers.

## Installation & Environment

1. Install dependencies with Poetry:
   ```bash
   poetry install --no-root
   ```
2. Use the virtual environment for commands:
   ```bash
   poetry run bitbat --help
   ```

## Configuration

- Default config is `src/bitbat/config/default.yaml` (`data_dir`, `freq`, `horizon`, `tau`, etc.).
- Override via `--config path/to/config.yaml` or the `BITBAT_CONFIG` environment variable.

## Streamlit UI

The Streamlit app provides the same workflow coverage as the CLI (ingest → features → model → predictions → monitor/backtest) and reads the same configuration values: `data_dir`, `freq`, `horizon`, and `tau`.

Run it with:
```bash
poetry run streamlit run streamlit/app.py
# or
make streamlit
```

Outputs are written to the same locations as the CLI:
- `${data_dir}/raw/` for ingested data
- `${data_dir}/features/{freq}_{horizon}/` for datasets + metadata
- `models/{freq}_{horizon}/` for trained models
- `${data_dir}/predictions/{freq}_{horizon}.parquet` for predictions
- `metrics/live_{freq}_{horizon}.json` for monitoring snapshots

## Data Ingestion

### Prices
```bash
poetry run bitbat prices pull --symbol BTC-USD --interval 1h --start 2017-01-01
```
- Writes partitioned parquet under `${data_dir}/raw/prices/` and enforces the prices contract.

### News
```bash
poetry run bitbat news pull --from 2024-01-01T00:00:00 --to 2024-01-02T00:00:00
```
- Writes GDELT-derived parquet under `${data_dir}/raw/news/gdelt_1h/`.

#### GDELT limits & price-only workflow
- GDELT ingestion is historical-only and capped at 30 days per pull (use incremental ranges). Default throttling/retry values are `news_throttle_seconds: 10.0` and `news_retry_limit: 30` from `src/bitbat/config/default.yaml`.
- Recommended realtime approach is price-only: disable sentiment **before** running `bitbat features build`, `bitbat model train`, and downstream steps by setting `enable_sentiment: false` in `src/bitbat/config/default.yaml` or unchecking the Streamlit "Enable sentiment features" checkbox.
- The CLI reads `enable_sentiment` from config for `bitbat features build`, `bitbat batch run`, and `bitbat model infer` (feature contract expectations follow the same toggle). The Streamlit ingest page is for historical GDELT pulls only.

## Feature Engineering
```bash
poetry run bitbat features build --start 2024-01-01T00:00:00 --end 2024-02-01T00:00:00
```
- Produces `${data_dir}/features/{freq}_{horizon}/dataset.parquet` plus `meta.json`.
- Features are `feat_*` columns; includes `label` and `r_forward` for training.

## Model Workflows

- **Cross-validation**:
  ```bash
  poetry run bitbat model cv --start 2024-01-01T00:00:00 --end 2024-02-01T00:00:00
  ```
  Uses walk-forward splits with embargo.

- **Training**:
  ```bash
  poetry run bitbat model train
  ```
  Loads the contract dataset, fits XGBoost, saves under `models/{freq}_{horizon}/`.

- **Inference**:
  ```bash
  poetry run bitbat model infer --features path/to/features.parquet --output predictions.parquet
  ```
  Validates feature contract, predicts per-row, outputs contract-compliant predictions.

## Batch Operations

- **Run**: Generates the latest prediction and appends/deduplicates.
  ```bash
  poetry run bitbat batch run
  ```

- **Realize**: Fills in realized returns/labels once the horizon has elapsed.
  ```bash
  poetry run bitbat batch realize --tau 0.01
  ```

## Backtesting & Monitoring

- **Backtest**:
  ```bash
  poetry run bitbat backtest run --enter-threshold 0.6 --allow-short
  ```

- **Monitor**:
  ```bash
  poetry run bitbat monitor refresh --cost-bps 4
  ```
  Writes live metrics to `${data_dir}/metrics/live_{freq}_{horizon}.json`.

## Testing & Quality Checks

- Lint: `poetry run ruff check src tests`
- Type check: `poetry run mypy src tests`
- Tests: `poetry run pytest`

Refer to [Testing & Quality](./testing-quality.md) for more guardrail details.
