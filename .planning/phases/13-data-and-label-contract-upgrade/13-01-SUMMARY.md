---
phase: 13-data-and-label-contract-upgrade
plan: "01"
subsystem: dataset
tags: [timealign, asof, contracts, leakage]
requires:
  - phase: 12-simplified-ui-regression-gates
    provides: Stable UI and release-gate baseline so v1.2 can focus on prediction data correctness
provides:
  - Leakage-safe as-of alignment helper for feature joins
  - Dataset build integration that enforces backward-only source alignment
  - Timestamp integrity contract checks for feature datasets
affects: [dataset-build, timealign, feature-contracts, v1.2-phase13]
tech-stack:
  added: []
  patterns:
    - Shared as-of alignment helper reused by dataset feature joins
    - Timestamp monotonicity and uniqueness treated as contract-level invariants
key-files:
  created:
    - src/bitbat/timealign/asof.py
    - tests/timealign/test_asof_join.py
  modified:
    - src/bitbat/dataset/build.py
    - src/bitbat/contracts.py
    - tests/contracts/test_contracts.py
    - tests/dataset/test_build_xy.py
key-decisions:
  - "Centralized as-of behavior in `timealign/asof.py` instead of duplicating merge logic in each feature source join."
  - "Enforced sorted and unique `timestamp_utc` requirements in feature contracts to catch leakage-prone datasets early."
patterns-established:
  - "Auxiliary feature sources should align through shared backward-only as-of joins."
  - "Dataset contracts should validate time-order invariants, not only column schemas."
requirements-completed: [DATA-01]
duration: 2 min
completed: 2026-02-25
---

# Phase 13 Plan 01: As-of Alignment Contract Summary

**Leakage-safe as-of alignment is now a hard dataset contract across auxiliary feature joins.**

## Performance

- **Duration:** 2 min
- **Started:** 2026-02-26T00:17:46+01:00
- **Completed:** 2026-02-26T00:19:08+01:00
- **Tasks:** 3
- **Files modified:** 6

## Accomplishments
- Added a reusable as-of time-alignment module with explicit future-match rejection guards.
- Routed dataset feature joins through backward-only as-of alignment for sentiment/macro/on-chain sources.
- Hardened feature-contract validation so unsorted or duplicate timestamps fail fast.

## Task Commits

Each task was committed atomically:

1. **Task 1: Add shared as-of alignment helpers with explicit future-data guards** - `6de3d60` (feat)
2. **Task 2: Route dataset feature joins through the as-of guard path** - `8920205` (feat)
3. **Task 3: Harden feature-contract checks around timestamp integrity** - `0b32da4` (fix)

## Files Created/Modified
- `src/bitbat/timealign/asof.py` - Shared backward-only as-of join and future-match validation helpers.
- `tests/timealign/test_asof_join.py` - Unit coverage for boundary-inclusive and irregular-gap as-of behavior.
- `src/bitbat/dataset/build.py` - Uses as-of helper for sentiment/macro/on-chain joins.
- `src/bitbat/contracts.py` - Enforces sorted and unique `timestamp_utc` in feature contracts.
- `tests/contracts/test_contracts.py` - Contract regression tests for timestamp ordering/uniqueness.
- `tests/dataset/test_build_xy.py` - Dataset assertions for monotonic and unique timestamps.

## Decisions Made
- Shared as-of helper is now the only join path for auxiliary feature sources in dataset build.
- Timestamp ordering/uniqueness moved from implicit assumptions to explicit contract checks.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 13 can proceed to shared return/direction labeling semantics (Plan 13-02).
- DATA-01 guardrails are in place for downstream label and evaluation work.

## Self-Check: PASSED

- `poetry run pytest tests/timealign/test_asof_join.py -q` -> 4 passed
- `poetry run pytest tests/dataset/test_build_xy.py -q -k "asof or leakage or build_xy"` -> 1 passed
- `poetry run pytest tests/dataset/test_build_xy.py tests/contracts/test_contracts.py -q -k "timestamp or contract or leakage"` -> 6 passed

---
*Phase: 13-data-and-label-contract-upgrade*
*Completed: 2026-02-25*
