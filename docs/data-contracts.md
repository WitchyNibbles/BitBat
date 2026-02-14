# Data Contracts

Contract enforcement lives in `bitbat.contracts`. Each validator normalizes and checks schemas before data is persisted or consumed.
Validators also return a canonical column set/order (extra columns are dropped).

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
  - `feat_*` numeric feature columns (price and optional sentiment derived)
  - `label` (`"up"`, `"down"`, `"flat"`) — present when `require_label=True`
  - `r_forward` (float) — forward return for training modes
- Contract enforces `feat_` prefix and numeric dtype.
- `require_features_full=True` enforces the full sentiment feature set:
  - windows: `1h`, `4h`, `24h`
  - suffixes per window: `mean`, `median`, `pos`, `neg`, `neu`, `count`, `decay`
  - expected columns: `feat_sent_1h_*`, `feat_sent_4h_*`, `feat_sent_24h_*` (21 columns total)
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

Helpers ensure timestamps are tz-naive UTC, probabilities numeric, and metadata string-typed.
`realized_r` is parsed with numeric coercion (`errors="coerce"`), so invalid values become `NaN`.
Deduplication is not performed by `ensure_predictions_contract`; it is done in pipeline steps such as `bitbat batch run`.

## Enforcement Points

- **Prices contract (`ensure_prices_contract`)**
  - `bitbat.ingest.prices.fetch_yf` validates prices before parquet write.

- **News contract (`ensure_news_contract`)**
  - `bitbat.ingest.news_gdelt.fetch` validates merged news before parquet write.

- **Feature contract (`ensure_feature_contract`)**
  - `bitbat.dataset.build.build_xy` validates the assembled dataset before persistence.
  - CLI dataset loader (`_load_feature_dataset`) validates training/CV datasets for `bitbat model train` and `bitbat model cv`.
  - `bitbat model infer` validates input feature parquet.
  - `bitbat batch run` validates generated feature rows before inference.

- **Predictions contract (`ensure_predictions_contract`)**
  - `bitbat model infer` validates generated predictions before output.
  - `bitbat batch run` validates new, existing, and merged prediction frames before persistence.
  - `bitbat batch realize` validates predictions before and after realization updates.
  - `bitbat backtest run` and `bitbat monitor refresh` validate prediction inputs before metric computation.

Violations raise `ContractError`, causing pipelines to stop rather than propagate invalid data. Use the validators when introducing new pipelines or data sources.
