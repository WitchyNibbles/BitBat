# Monitoring Strategy

## Purpose
Define how BitBat monitors model quality in production, detects drift, and safely retrains with minimal manual intervention.

## Performance Metrics

### Primary
- `hit_rate`: fraction of realized predictions where predicted direction matches actual direction
- `sharpe_ratio`: mean realized return divided by return volatility
- `average_return`: mean realized return over evaluation window
- `max_drawdown`: largest peak-to-trough decline in cumulative return curve

### Stability
- `win_streak`: longest consecutive run of correct predictions
- `lose_streak`: longest consecutive run of incorrect predictions
- `current_streak`: active streak and sign (`win` or `loss`)

### Calibration
- `high_confidence_count`: number of realized predictions with max class probability >= 0.60
- `high_confidence_accuracy`: accuracy on high-confidence predictions
- `calibration_error`: absolute difference between mean confidence and high-confidence accuracy

## Drift Detection Rules

Drift is checked on a rolling window (`autonomous.drift_detection.window_days`, default 30 days) and only when at least `min_predictions_required` realized predictions exist.

Drift triggers when any of the following conditions hold:
- Hit-rate degradation:
  - current hit rate < 55%
  - and drop from baseline exceeds `hit_rate_drop_threshold` (default 5%)
- Risk deterioration:
  - Sharpe ratio < configured threshold (`sharpe_threshold`, default -0.5)
- Persistent failure:
  - losing streak >= 10
- Confidence mismatch:
  - high-confidence accuracy < 60% when enough high-confidence samples are available

## Baseline Definition

Baseline hit-rate proxy is taken from active model CV score when available (`model_versions.cv_score`). If unavailable, a conservative fallback baseline of 0.55 is used.

## Retraining Workflow

1. Monitoring loop validates pending predictions.
2. Latest realized predictions are converted to performance metrics.
3. Snapshot is stored in `performance_snapshots`.
4. Drift detector evaluates trigger rules.
5. If drift is detected and not in cooldown:
   - create `retraining_events` row (`status=started`)
   - run feature build, model CV, and model train commands
   - evaluate CV improvement threshold (`cv_improvement_threshold`)
   - deploy only when improvement threshold is met
   - update event to `completed` or `failed`

## Safety Mechanisms

- Cooldown enforcement:
  - retraining blocked for `autonomous.retraining.cooldown_hours` after last retraining start
- No-data protection:
  - drift checks are skipped when realized sample size is below threshold
- Command failure isolation:
  - retraining step failures are captured and logged; monitoring loop continues next cycle
- Partial failure tolerance:
  - one failed validation/update does not stop batch processing
- Explicit deployment gate:
  - model deployment requires minimum CV improvement

## Alerting Policy

- `INFO`: validation summary, monitoring loop successful
- `WARNING`: drift detected but cooldown active, or calibration concerns
- `CRITICAL`: retraining started
- `SUCCESS`: retraining completed and deployed
- `ERROR`: retraining failed or monitoring loop crashed

Alerts can route to:
- SMTP email
- Discord webhook
- Telegram bot API

## Execution Model

- Manual test run: `poetry run bitbat monitor run-once`
- Continuous mode: `poetry run bitbat monitor start --interval 3600`
- Service mode: `scripts/run_monitoring_agent.py` via systemd or container process manager
