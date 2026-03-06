---
phase: 25-critical-correctness-remediation
plan: 03
subsystem: testing, api
tags: [pr-auc, leakage, guardrail, xgboost, api-defaults, config]

# Dependency graph
requires:
  - phase: 24-audit-baseline
    provides: "Audit findings 17 (missing test_leakage.py) and 18 (hardcoded API defaults)"
provides:
  - "PR-AUC guardrail test detecting train/test information leakage"
  - "No-future-timestamps test verifying feature computation integrity"
  - "OBV no-lookahead test"
  - "Config-sourced API route defaults (freq/horizon from default.yaml)"
  - "Structural guard test preventing re-hardcoding of API defaults"
affects: [api, features, monitoring]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Config-sourced defaults via api/defaults.py helper, computed once at import time"
    - "PR-AUC guardrail pattern: train on random labels, assert low score"

key-files:
  created:
    - tests/features/test_leakage.py
    - src/bitbat/api/defaults.py
    - tests/api/test_api_config_defaults.py
  modified:
    - src/bitbat/api/routes/predictions.py
    - src/bitbat/api/routes/analytics.py
    - src/bitbat/api/routes/health.py
    - tests/api/test_phase4_complete.py

key-decisions:
  - "PR-AUC threshold set at 0.7 (random labels should yield ~0.5, so 0.7 gives margin while catching leakage)"
  - "API defaults computed once at module import via _default_freq/_default_horizon rather than per-request to avoid repeated YAML parsing"
  - "Fallback values '1h'/'4h' kept in defaults.py .get() for the case where config file is completely broken"

patterns-established:
  - "Config-sourced defaults pattern: api/defaults.py helper -> module-level _FREQ/_HORIZON -> Query(_FREQ) in routes"
  - "Structural guard test: read source files and assert no hardcoded strings (prevents regression)"

requirements-completed: [CORR-05, CORR-06]

# Metrics
duration: 7min
completed: 2026-03-07
---

# Phase 25 Plan 03: Leakage Guardrail Test and Config-Sourced API Defaults Summary

**PR-AUC guardrail test for feature leakage detection (CORR-05) and config-sourced freq/horizon defaults in all API routes (CORR-06)**

## Performance

- **Duration:** 7 min
- **Started:** 2026-03-06T23:15:00Z
- **Completed:** 2026-03-06T23:22:18Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments
- Created test_leakage.py with 3 tests: PR-AUC guardrail, no-future-timestamps, OBV no-lookahead -- making the CLAUDE.md reference to this file accurate
- Replaced all hardcoded Query("1h")/Query("4h") defaults in predictions, analytics, and health routes with config-sourced values from default.yaml (currently 5m/30m)
- Added structural guard test that reads API route source files to prevent re-hardcoding

## Task Commits

Each task was committed atomically:

1. **Task 1: Create test_leakage.py with PR-AUC guardrail (CORR-05)** - `7099eae` (test)
2. **Task 2: Source API freq/horizon defaults from config and add test (CORR-06)** - `d316ba4` (fix)

## Files Created/Modified
- `tests/features/test_leakage.py` - PR-AUC guardrail, no-future-timestamps, OBV no-lookahead tests
- `src/bitbat/api/defaults.py` - Shared helper for config-sourced freq/horizon defaults
- `tests/api/test_api_config_defaults.py` - 3 tests verifying API defaults match config
- `src/bitbat/api/routes/predictions.py` - Replaced hardcoded 1h/4h with _FREQ/_HORIZON from config
- `src/bitbat/api/routes/analytics.py` - Replaced hardcoded 1h/4h with _FREQ/_HORIZON from config
- `src/bitbat/api/routes/health.py` - Replaced hardcoded 1h/4h in _check_model/_check_dataset defaults
- `tests/api/test_phase4_complete.py` - Updated fixture to create files at config-default paths

## Decisions Made
- PR-AUC threshold set at 0.7: random labels should yield ~0.5, so 0.7 gives comfortable margin while still catching genuine leakage
- Created `api/defaults.py` as a shared module rather than duplicating config reads across route files
- Module-level constants (`_FREQ`, `_HORIZON`) computed once at import time to avoid repeated YAML parsing on every request
- Kept `"1h"`/`"4h"` as last-resort `.get()` fallbacks in defaults.py for the edge case where config is completely broken

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed test_phase4_complete fixture for config-default paths**
- **Found during:** Task 2 (API defaults update)
- **Issue:** After changing health endpoint defaults from hardcoded "1h"/"4h" to config-sourced "5m"/"30m", the test_detailed_health_all_ok test failed because the fixture only created model/dataset files at the `1h_4h` path, not the config-default `5m_30m` path
- **Fix:** Updated the `full_env` fixture to also create model and dataset files at the config-default freq/horizon path when it differs from "1h_4h"
- **Files modified:** tests/api/test_phase4_complete.py
- **Verification:** Full test suite (619 tests) passes with 0 failures
- **Committed in:** d316ba4 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Direct consequence of the API defaults change. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- CORR-05 and CORR-06 are complete
- CLAUDE.md reference to test_leakage.py is now accurate
- API routes now respect default.yaml configuration
- Ready for 25-04 (remaining correctness items)

## Self-Check: PASSED

All 7 created/modified files verified on disk. Both task commits (7099eae, d316ba4) found in git log.

---
*Phase: 25-critical-correctness-remediation*
*Completed: 2026-03-07*
