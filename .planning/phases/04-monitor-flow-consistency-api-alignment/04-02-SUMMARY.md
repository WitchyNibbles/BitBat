---
phase: 04-monitor-flow-consistency-api-alignment
plan: "02"
subsystem: api
tags: [fastapi, prediction-api, widgets, streamlit, semantics]
requires:
  - phase: 04-01
    provides: Canonical monitor prediction semantics for return/price/correctness
provides:
  - API prediction responses aligned to monitor return/price semantics
  - GUI widget read model/rendering aligned to nullable prediction price semantics
  - Cross-surface regression coverage using ASGI transport client compatible with current runtime
affects: [phase-05, timeline-read-model, api-consumers, gui-home]
tech-stack:
  added: []
  patterns: [async-fastapi-handlers, asgi-transport-sync-wrapper, cross-surface-semantic-regressions]
key-files:
  created:
    - .planning/phases/04-monitor-flow-consistency-api-alignment/04-02-SUMMARY.md
    - tests/api/client.py
  modified:
    - src/bitbat/api/routes/predictions.py
    - src/bitbat/api/routes/analytics.py
    - src/bitbat/api/routes/health.py
    - src/bitbat/api/routes/metrics.py
    - src/bitbat/gui/widgets.py
    - tests/api/test_predictions.py
    - tests/api/test_phase4_complete.py
    - tests/api/test_health.py
    - tests/api/test_metrics.py
    - tests/gui/test_widgets.py
key-decisions:
  - "Expose return/price semantics as primary API prediction fields and compute directional metrics directly from predicted vs actual returns."
  - "Replace FastAPI TestClient usage in API suites with a lightweight ASGI transport client due runtime anyio portal/threadpool hangs."
patterns-established:
  - "API route handlers are async to avoid blocked anyio threadpool paths in this runtime."
  - "API integration tests use SyncASGIClient (httpx ASGI transport) for deterministic endpoint verification."
requirements-completed: [MON-02, API-01]
duration: 18 min
completed: 2026-02-24
---

# Phase 04 Plan 02: API and GUI Semantic Alignment Summary

**API prediction payloads and GUI latest-prediction widgets now share the same return/price semantic contract with cross-surface regressions.**

## Performance

- **Duration:** 18 min
- **Started:** 2026-02-24T14:20:00Z
- **Completed:** 2026-02-24T14:38:00Z
- **Tasks:** 3
- **Files modified:** 10

## Accomplishments
- Aligned prediction API mapping to return/price semantics and extended performance output with directional/error metrics.
- Updated widget read/render behavior to display explicit predicted return/price semantics instead of implicit confidence fallbacks.
- Added cross-surface regression coverage with a stable ASGI transport test client for prediction, health, metrics, and phase-4 integration suites.

## Task Commits

1. **Task 1: Expand API prediction read model to expose aligned semantic fields** - `4162d42` (feat)
2. **Task 2: Align GUI timeline and widget read helpers with nullable confidence semantics** - `ecfcd61` (feat)
3. **Task 3: Add cross-surface semantic consistency regression tests** - `b43ab0b` (test)

## Files Created/Modified
- `.planning/phases/04-monitor-flow-consistency-api-alignment/04-02-SUMMARY.md` - Plan summary and execution traceability.
- `tests/api/client.py` - Sync ASGI transport wrapper used by API tests in this runtime.
- `src/bitbat/api/routes/predictions.py` - Prediction response mapping and performance metric enrichment aligned to monitor semantics.
- `src/bitbat/api/routes/analytics.py` - Async route handler conversion for transport compatibility.
- `src/bitbat/api/routes/health.py` - Async route handler conversion for transport compatibility.
- `src/bitbat/api/routes/metrics.py` - Async route handler conversion for transport compatibility.
- `src/bitbat/gui/widgets.py` - Latest prediction contract/rendering updated for predicted return/price semantics.
- `tests/api/test_predictions.py` - Prediction endpoint regression coverage aligned to new contract.
- `tests/api/test_phase4_complete.py` - End-to-end phase integration assertions aligned to updated semantics.
- `tests/api/test_health.py` - Health endpoint coverage migrated to SyncASGIClient.
- `tests/api/test_metrics.py` - Metrics endpoint coverage migrated to SyncASGIClient.
- `tests/gui/test_widgets.py` - Widget contract regressions updated for return/price semantics.

## Decisions Made
- Kept API compatibility additive: return/price semantics are exposed without removing existing endpoint surfaces.
- Treated runtime client hang as a blocking infrastructure issue and standardized on ASGI transport tests for deterministic execution.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] FastAPI TestClient hang in current runtime**
- **Found during:** Task 3 (cross-surface regression verification)
- **Issue:** `TestClient`/anyio portal and threadpool paths hung consistently, blocking API suite execution.
- **Fix:** Converted API handlers to async definitions and introduced `SyncASGIClient` (httpx ASGI transport wrapper) for API test modules.
- **Files modified:** `src/bitbat/api/routes/{predictions,analytics,health,metrics}.py`, `tests/api/client.py`, API test modules
- **Verification:** API/GUI regression commands complete successfully with expected pass counts.
- **Committed in:** `b43ab0b` (Task 3 commit, plus async route support from `4162d42`)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Required to execute planned verification in this runtime; no product-scope expansion.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 5 timeline reliability work can consume API/widget records with consistent return/price semantics.
- Cross-surface regressions are in place to catch semantic drift between monitor persistence, API responses, and widget presentation.

## Self-Check: PASSED

- Verified key files from summary metadata exist on disk.
- Verified `git log --grep="04-02"` contains task commits.
- Verified targeted API/GUI semantic regression suites pass.

---
*Phase: 04-monitor-flow-consistency-api-alignment*
*Completed: 2026-02-24*
