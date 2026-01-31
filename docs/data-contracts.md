# Data Contracts

Contract enforcement lives in `alpha.contracts`. Each validator normalizes and checks schemas before data is persisted or consumed.

## Prices Parquet

Enforced by `ensure_prices_contract`:

| Column         | Type              | Notes                         |
|----------------|-------------------|-------------------------------|
| `timestamp_utc`| `datetime64[ns]`   | Bar end, UTC, tz-naive        |
| `open`         | `float64`         | OHLCV open                    |
| `high`         | `float64`         | OHLCV high                    |
| `low`          | `float64`         | OHLCV low                     |
| `close`        | `float64`         | OHLCV close                   |
| `volume`       | `float64`         | Raw volume                    |
| `source`       | `string`          | e.g., `"yfinance"`           |

## News Parquet

Enforced by `ensure_news_contract`:

| Column           | Type              | Notes                                  |
|------------------|-------------------|----------------------------------------|
| `published_utc`  | `datetime64[ns]`   | Publication time, UTC, tz-naive        |
| `title`          | `string`          | Headline text                          |
| `url`            | `string`          | Canonical URL (dedupe key)             |
| `source`         | `string`          | Publisher name                         |
| `lang`           | `string`          | ISO-639-1 if available                 |
| `sentiment_score`| `float64`         | VADER compound score                   |

## Feature Dataset Parquet

Produced by `build_xy` and validated via `ensure_feature_contract`:

- Columns:
  - `timestamp_utc` (datetime, UTC, tz-naive)
  - `feat_*` numeric feature columns (price & sentiment derived)
  - `label` (`"up"`, `"down"`, `"flat"`) — present when `require_label=True`
  - `r_forward` (float) — forward return for training modes
- Contract enforces `feat_` prefix and numeric dtype.
- Persisted metadata (`meta.json`) captures `freq`, `horizon`, `tau`, `seed`, `version`, and class counts.

## Predictions Parquet

Validated by `ensure_predictions_contract`:

| Column            | Type              | Notes                                      |
|-------------------|-------------------|--------------------------------------------|
| `timestamp_utc`   | `datetime64[ns]`   | Bar timestamp associated with prediction   |
| `p_up`            | `float64`         | Probability of upward move                 |
| `p_down`          | `float64`         | Probability of downward move               |
| `horizon`         | `string`          | e.g., `"4h"`                              |
| `freq`            | `string`          | e.g., `"1h"`                              |
| `model_version`   | `string`          | Version tag (defaults to package version)  |
| `realized_r`      | `float64`         | Realized return (NaN until realized)       |
| `realized_label`  | `string`          | Realized directional label (nullable)      |

Helpers ensure timestamps are tz-naive UTC, probabilities numeric, metadata string-typed, and duplicates removed via natural key (`timestamp_utc`, `horizon`, `model_version`).

## Enforcement Points

- Ingestion modules call contracts before writing parquet (prices, news).
- `build_xy` ensures features obey contract prior to persistence and downstream consumption.
- CLI flows (`model infer`, `batch run/realize`, `backtest`, `monitor`) validate predictions after read/write cycles.

Violations raise `ContractError`, causing pipelines to stop rather than propagate invalid data. Use the validators when introducing new pipelines or data sources.
