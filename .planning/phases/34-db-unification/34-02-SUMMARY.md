---
phase: 34-db-unification
plan: 02
subsystem: api-system
tags: [api, fastapi, database, tdd, fail-fast]

requires:
  - phase: 34-01
    provides: AutonomousDB read facade and DB-layer failure handling
provides:
  - /system logs/events/snapshots routed through AutonomousDB
  - fail-fast API DB errors with short message plus hint line
  - direct regression coverage for /system DB-backed routes
affects: [34-03, api, monitoring]

tech-stack:
  added: []
  patterns:
    - thin FastAPI routes over AutonomousDB payload methods
    - standardized DB-to-HTTP 503 conversion with hint line

key-files:
  created:
    - tests/api/test_system.py
  modified:
    - src/bitbat/api/routes/system.py

key-decisions:
  - "The DB-backed /system routes fail fast with HTTP 503 instead of returning degraded empty payloads"
  - "Route handlers delegate schema compatibility and lock handling to AutonomousDB rather than re-implementing raw SQL fallback logic"

patterns-established:
  - "DB-backed API routes should construct a short user-facing detail plus Hint line from MonitorDatabaseError"

requirements-completed: [DEBT-03]

duration: 10min
completed: 2026-03-12
---

# Phase 34 Plan 02: DB Unification Summary

**The `/system` API routes now use AutonomousDB instead of direct sqlite access, with dedicated regression coverage for both happy-path payloads and fail-fast DB errors**

## Performance

- **Duration:** 10 min
- **Completed:** 2026-03-12T16:08:04Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Added [test_system.py](/home/eimi/projects/ai-btc-predictor/tests/api/test_system.py) to lock route behavior to `AutonomousDB` payloads and explicit DB failure handling
- Replaced the raw sqlite helpers in [system.py](/home/eimi/projects/ai-btc-predictor/src/bitbat/api/routes/system.py) with `AutonomousDB` calls for logs, retraining events, and performance snapshots
- Standardized request failure formatting to `short message + Hint: ...` without exposing low-level exception names

## Task Commits

1. **Task 1: Add failing API tests for unified /system DB reads** - `160524c` (test)
2. **Task 2: Migrate /system DB routes to AutonomousDB and standardize failure handling** - `6dbf2ad` (feat)

## Verification

- `poetry run pytest tests/api/test_system.py tests/api/test_settings.py tests/api/test_no_gui_import.py -x`
- `poetry run ruff check src/bitbat/api/routes/system.py tests/api/test_system.py`

## Decisions Made

- Kept settings and training endpoints untouched; only the DB-backed read routes moved to the unified layer
- Converted DB-layer failures to HTTP 503 with a single hint line, matching the Phase 34 diagnostics decision without leaking `MonitorDatabaseError.error_class`

## Deviations from Plan

None.

## Next Phase Readiness

- Plan 34-02 is complete and verified
- The remaining raw sqlite runtime call sites are now limited to the Streamlit helpers targeted by Plan 34-03
- API route coverage is in place to catch regressions while the remaining GUI/retrainer migrations land

---
*Phase: 34-db-unification*
*Completed: 2026-03-12*
