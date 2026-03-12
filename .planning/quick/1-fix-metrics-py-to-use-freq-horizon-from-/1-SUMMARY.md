---
quick_task: 1
description: fix metrics.py to use _FREQ/_HORIZON from api/defaults.py
date: 2026-03-07
commit: c03721f
status: complete
---

# Quick Task 1: Fix metrics.py hardcoded 1h/4h

## What was done

Replaced three hardcoded `"1h"`/`"4h"` literals in `src/bitbat/api/routes/metrics.py`
with module-level `_FREQ`/`_HORIZON` constants sourced from `api/defaults.py` —
matching the pattern already applied to `predictions.py`, `analytics.py`, and `health.py`
in phase 25-03.

## Changes

**`src/bitbat/api/routes/metrics.py`**
- Added import: `from bitbat.api.defaults import _default_freq, _default_horizon`
- Added module-level constants: `_FREQ = _default_freq()`, `_HORIZON = _default_horizon()`
- Line 76: `Path("models/1h_4h/xgb.json")` → `Path(f"models/{_FREQ}_{_HORIZON}/xgb.json")`
- Line 82: `Path("data/features/1h_4h/dataset.parquet")` → `Path(f"data/features/{_FREQ}_{_HORIZON}/dataset.parquet")`
- Lines 94-95: `db.get_recent_predictions(session, "1h", "4h", ...)` → `db.get_recent_predictions(session, _FREQ, _HORIZON, ...)`

## Verification

- 63 API tests pass, 0 failures
- Commit: c03721f

## Root cause

This route was missed when phase 25-03 updated the other three API route files.
The structural guard test in `tests/api/test_api_config_defaults.py` only checked
`Query("1h")` patterns (route parameter defaults), not positional args to DB calls
or `Path()` constructors, so it didn't catch this instance.
