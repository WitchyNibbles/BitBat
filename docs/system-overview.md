# System Overview

BitBat is an end-to-end pipeline for generating directional forecasts on Bitcoin price movements. It ingests market and news data, engineers aligned feature sets, trains gradient-boosted decision tree models, evaluates performance with leakage guardrails, and runs batch inference/monitoring jobs for live predictions.

## Core Flow

1. **Ingestion**
   - `bitbat.ingest.prices.fetch_yf` downloads hourly OHLCV bars from Yahoo Finance (yfinance), enforcing a schema contract.
   - `bitbat.ingest.news_gdelt.fetch` queries the GDELT API to obtain crypto-related news headlines and sentiment scores.

2. **Feature Engineering**
   - `bitbat.features.price` derives lagged returns, volatility, ATR, MACD, and OBV features.
   - `bitbat.features.sentiment.aggregate` produces sentiment aggregates aligned to bar timestamps with look-ahead protection.

3. **Dataset Assembly**
   - `bitbat.dataset.build.build_xy` merges price/news features, assigns directional labels via forward returns and tau thresholds, and persists a contract-compliant dataset with metadata (freq, horizon, tau, seed, version).

4. **Model Training & Evaluation**
   - `bitbat.model.train.fit_xgb` fits a multi-class XGBoost classifier with optional class weighting, storing models per freq/horizon.
   - `bitbat.model.evaluate.classification_metrics` computes robust metrics, writes artifacts, and warns on suspicious PR-AUC jumps to detect leakage.

5. **CLI Workflows**
   - `bitbat/cli.py` (Click-based) exposes command groups for ingestion, feature builds, model CV/train/infer, backtest, batch operations, and monitoring.

6. **Batch & Monitoring**
   - `bitbat.cli.batch_run` generates new predictions, persists them with schema validation, and avoids duplicates via natural keys.
   - `bitbat.cli.batch_realize` fills in realized returns/labels once horizon data arrives.
   - `bitbat.cli.monitor_refresh` summarizes recent live performance metrics.

## Guardrails & Contracts

- **Contracts**: `bitbat.contracts` centralizes schema enforcement for prices, news, features, and predictions.
- **Time alignment**: Feature aggregation and dataset assembly respect timestamp ordering; tests ensure no future leakage.
- **Evaluation warning**: PR-AUC jumps >10 pts with unchanged class balance emit console alerts.
- **Idempotency**: Ingestion and batch flows deduplicate using natural keys.
- **Embargo**: Walk-forward splits apply embargo bars between train/test windows to prevent leakage.

Refer to the [Project Structure](./project-structure.md) and [Usage Guide](./usage-guide.md) for implementation details and operational instructions.
