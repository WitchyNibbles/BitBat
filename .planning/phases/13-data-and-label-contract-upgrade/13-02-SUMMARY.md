---
phase: 13-data-and-label-contract-upgrade
plan: "02"
subsystem: labeling
tags: [returns, targets, horizon, contracts]
requires:
  - phase: 13-data-and-label-contract-upgrade
    provides: Leakage-safe as-of feature alignment and timestamp integrity checks from Plan 13-01
provides:
  - Canonical horizon parsing and deterministic forward-return APIs
  - Direction labels derived from the same forward-return path
  - Dataset contract wiring that persists synchronized `label` and `r_forward` outputs
affects: [labeling, dataset-build, feature-contracts, v1.2-phase13]
tech-stack:
  added: []
  patterns:
    - Direction labels are always computed from canonical forward-return outputs
    - Dataset contract validation enforces non-null, bounded direction labels when required
key-files:
  created: []
  modified:
    - src/bitbat/labeling/returns.py
    - src/bitbat/labeling/targets.py
    - src/bitbat/labeling/__init__.py
    - src/bitbat/dataset/build.py
    - src/bitbat/contracts.py
    - tests/labeling/test_returns.py
    - tests/labeling/test_targets.py
    - tests/dataset/test_build_xy.py
key-decisions:
  - "Introduced `forward_return_from_close` + `parse_horizon` so all downstream label paths consume one horizon parser and return generator."
  - "Routed direction labeling through `direction_from_prices`/`direction_from_returns` instead of maintaining a parallel horizon implementation."
  - "Updated dataset build to persist `label` with `r_forward` and require both through feature contracts."
patterns-established:
  - "Return and direction labels must be generated from one shared horizon-aware labeling path."
  - "Feature-contract checks enforce allowed direction classes (`up`, `down`, `flat`) when labels are required."
requirements-completed: [DATA-02]
duration: 8 min
completed: 2026-02-25
---

# Phase 13 Plan 02: Shared Return and Direction Label Contract Summary

**Return targets and direction labels now share a single horizon-aware labeling contract across helpers and dataset build output.**

## Performance

- **Duration:** 8 min
- **Tasks:** 3
- **Files modified:** 8

## Accomplishments

- Added canonical horizon parsing and forward-return computation helpers with deterministic index validation.
- Introduced direction-label APIs that derive directly from canonical forward returns (no duplicate horizon path).
- Integrated synchronized `label` + `r_forward` generation into `build_xy` and contract validation.

## Task Commits

Each task was committed atomically:

1. **Task 1: Introduce a canonical horizon and return-target labeling interface** - `fd51a85` (feat)
2. **Task 2: Derive direction labels directly from canonical forward returns** - `acc9deb` (feat)
3. **Task 3: Integrate the shared label contract into dataset build and validation** - `614c435` (feat)

## Files Created/Modified

- `src/bitbat/labeling/returns.py` - canonical horizon parsing + deterministic forward-return utilities.
- `src/bitbat/labeling/targets.py` - direction labels generated from shared return outputs.
- `src/bitbat/labeling/__init__.py` - exports canonical return helpers for package-level reuse.
- `src/bitbat/dataset/build.py` - writes aligned `label` + `r_forward` targets from one label frame.
- `src/bitbat/contracts.py` - validates label non-nullness and allowed direction vocabulary when labels required.
- `tests/labeling/test_returns.py` - coverage for parse behavior and index determinism.
- `tests/labeling/test_targets.py` - coverage for price-to-return-to-direction consistency.
- `tests/dataset/test_build_xy.py` - dataset-level alignment checks between `r_forward` and `label`.

## Decisions Made

- Kept return target (`r_forward`) as the primary training signal while persisting direction labels for aligned downstream use.
- Preserved backward-compatible `build_xy` return signature (`X`, `y`, `meta`) while strengthening persisted dataset schema.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None.

## Next Phase Readiness

- Plan 13-03 can add optional triple-barrier labeling mode on top of this shared baseline contract.
- DATA-02 is now implementation- and test-complete.

## Self-Check: PASSED

- `poetry run pytest tests/labeling/test_returns.py -q` -> 7 passed
- `poetry run pytest tests/labeling/test_targets.py -q` -> 5 passed
- `poetry run pytest tests/dataset/test_build_xy.py tests/labeling/test_returns.py tests/labeling/test_targets.py -q -k "label or direction or horizon"` -> 12 passed, 1 deselected
- `poetry run pytest tests/contracts/test_contracts.py -q` -> 6 passed

---
*Phase: 13-data-and-label-contract-upgrade*
*Completed: 2026-02-25*
