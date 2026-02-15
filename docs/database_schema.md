# Database Schema Design

This document defines the autonomous monitoring database schema used to track
predictions, model versions, retraining activity, performance snapshots, and
system logs.

## `prediction_outcomes`

```sql
CREATE TABLE IF NOT EXISTS prediction_outcomes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp_utc DATETIME NOT NULL,
  prediction_timestamp DATETIME NOT NULL,
  predicted_direction TEXT NOT NULL CHECK (predicted_direction IN ('up', 'down', 'flat')),
  p_up REAL NOT NULL,
  p_down REAL NOT NULL,
  p_flat REAL,
  predicted_return REAL,
  actual_return REAL,
  actual_direction TEXT CHECK (actual_direction IN ('up', 'down', 'flat')),
  correct BOOLEAN,
  model_version TEXT NOT NULL,
  freq TEXT NOT NULL,
  horizon TEXT NOT NULL,
  features_used JSON,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  realized_at DATETIME
);
```

Indexes:
- `idx_timestamp` on `timestamp_utc`
- `idx_model_version` on `model_version`
- `idx_freq_horizon` on `(freq, horizon)`
- `idx_unrealized` on `actual_return` where `actual_return IS NULL`
- `idx_created_at` on `created_at`

Expected JSON structure (`features_used`):

```json
{
  "feat_ret_1": 0.0012,
  "feat_vol_24": 0.0185,
  "feat_sent_1h_mean": 0.33
}
```

## `model_versions`

```sql
CREATE TABLE IF NOT EXISTS model_versions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  version TEXT UNIQUE NOT NULL,
  freq TEXT NOT NULL,
  horizon TEXT NOT NULL,
  training_start DATETIME NOT NULL,
  training_end DATETIME NOT NULL,
  training_samples INTEGER NOT NULL,
  cv_score REAL,
  features JSON,
  hyperparameters JSON,
  deployed_at DATETIME,
  replaced_at DATETIME,
  is_active BOOLEAN DEFAULT TRUE,
  training_metadata JSON,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

Indexes:
- `idx_version` on `version`
- `idx_active` on `is_active`
- `idx_freq_horizon_mv` on `(freq, horizon)`

Expected JSON structures:

`features`:

```json
["feat_ret_1", "feat_vol_24", "feat_sent_1h_mean"]
```

`hyperparameters`:

```json
{
  "n_estimators": 500,
  "max_depth": 6,
  "eta": 0.05,
  "subsample": 0.8
}
```

`training_metadata`:

```json
{
  "seed": 42,
  "tau": 0.0015,
  "class_weights": {
    "up": 1.0,
    "down": 1.1,
    "flat": 0.9
  }
}
```

## `retraining_events`

```sql
CREATE TABLE IF NOT EXISTS retraining_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  trigger_reason TEXT NOT NULL CHECK (trigger_reason IN ('drift_detected', 'scheduled', 'manual', 'poor_performance')),
  trigger_metrics JSON,
  old_model_version TEXT,
  new_model_version TEXT,
  cv_improvement REAL,
  training_duration_seconds REAL,
  status TEXT NOT NULL CHECK (status IN ('started', 'completed', 'failed')),
  error_message TEXT,
  started_at DATETIME NOT NULL,
  completed_at DATETIME
);
```

Indexes:
- `idx_started_at` on `started_at`
- `idx_status` on `status`

Expected JSON structure (`trigger_metrics`):

```json
{
  "window_days": 30,
  "hit_rate": 0.47,
  "hit_rate_baseline": 0.55,
  "sharpe_ratio": -0.62
}
```

## `performance_snapshots`

```sql
CREATE TABLE IF NOT EXISTS performance_snapshots (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  model_version TEXT NOT NULL,
  freq TEXT NOT NULL,
  horizon TEXT NOT NULL,
  snapshot_time DATETIME NOT NULL,
  window_days INTEGER NOT NULL,
  total_predictions INTEGER NOT NULL,
  realized_predictions INTEGER NOT NULL,
  hit_rate REAL,
  sharpe_ratio REAL,
  avg_return REAL,
  max_drawdown REAL,
  win_streak INTEGER,
  lose_streak INTEGER,
  calibration_score REAL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

Indexes:
- `idx_snapshot_time` on `snapshot_time`
- `idx_model_version_ps` on `model_version`
- `idx_freq_horizon_ps` on `(freq, horizon)`

No JSON columns in this table.

## `system_logs`

```sql
CREATE TABLE IF NOT EXISTS system_logs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  level TEXT NOT NULL CHECK (level IN ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')),
  service TEXT NOT NULL,
  message TEXT NOT NULL,
  details JSON,
  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

Indexes:
- `idx_timestamp_sl` on `timestamp`
- `idx_level` on `level`
- `idx_service` on `service`

Expected JSON structure (`details`):

```json
{
  "model_version": "0.1.0-20260214",
  "freq": "1h",
  "horizon": "4h",
  "exception": "TimeoutError"
}
```
