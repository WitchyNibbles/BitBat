---
phase: 16-promotion-guardrails-and-optimization-safety
plan: "01"
subsystem: model-optimization
tags: [nested-validation, optuna, provenance, cli, evaluation-safety]
requires:
  - phase: 15-cost-aware-walk-forward-evaluation
    provides: Candidate reporting/champion artifacts and cost-aware fold metrics from Phase 15
provides:
  - Nested walk-forward optimization where inner folds tune and outer folds score out-of-sample
  - Deterministic optimization provenance payloads with fold windows, trial history, and best-trial lineage
  - CLI `model optimize` workflow with config-driven controls and persisted optimization summary artifact
affects: [model-optimize, model-cv, cli, config, v1.2-phase16]
tech-stack:
  added: []
  patterns:
    - Nested optimization outputs must separate tuning folds from evaluation folds by construction
    - Optimization artifacts must be reproducible and audit-friendly for promotion safety reviews
key-files:
  created: []
  modified:
    - src/bitbat/model/optimize.py
    - src/bitbat/cli.py
    - src/bitbat/config/default.yaml
    - tests/model/test_optimize.py
    - tests/test_cli.py
key-decisions:
  - "Made nested walk-forward optimization the default mode and selected final params from best outer-fold score."
  - "Kept provenance deterministic by persisting structured metadata while avoiding non-deterministic runtime timestamps."
patterns-established:
  - "Optimization summaries must include outer_folds and provenance keys for downstream safeguards."
  - "CLI optimization controls resolve from model.optimization config first, then explicit flags."
requirements-completed: [EVAL-04]
duration: 6 min
completed: 2026-02-26
---

# Phase 16 Plan 01: Nested Optimization and Provenance Summary

**BitBat now runs nested walk-forward hyperparameter optimization with deterministic provenance and a dedicated CLI artifact path for auditable model tuning.**

## Performance

- **Duration:** 6 min
- **Started:** 2026-02-26T09:11:49+01:00
- **Completed:** 2026-02-26T09:14:29+01:00
- **Tasks:** 3
- **Files modified:** 5

## Accomplishments

- Implemented nested optimization in `HyperparamOptimizer` with inner-fold tuning and held-out outer-fold scoring.
- Expanded optimization summary/provenance payloads with fold windows, trial history, and best-trial lineage metadata.
- Added `bitbat model optimize` CLI command and persisted `metrics/optimization_summary.json` with config + provenance details.

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement nested walk-forward optimization engine (inner tune, outer evaluate)** - `79388ba` (feat)
2. **Task 2: Persist optimization provenance artifacts for auditability** - `6dc6058` (feat)
3. **Task 3: Expose nested optimization controls through CLI/config** - `f1b7da1` (feat)

## Files Created/Modified

- `src/bitbat/model/optimize.py` - nested outer/inner optimization flow and deterministic provenance serialization.
- `src/bitbat/cli.py` - `model optimize` command, config resolution, and optimization summary persistence.
- `src/bitbat/config/default.yaml` - default optimization controls for nested trial execution.
- `tests/model/test_optimize.py` - nested optimization metadata/provenance regression coverage.
- `tests/test_cli.py` - CLI nested optimization artifact persistence regression test.

## Decisions Made

- Used per-outer-fold seeded Optuna studies so nested search stays deterministic across repeated runs.
- Added explicit `outer_folds` and `provenance` structures to optimization summaries to standardize downstream safeguard consumption.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None.

## Next Phase Readiness

- Nested validation and provenance foundations for EVAL-04 are now implemented and tested.
- Phase 16 Plan 02 can integrate multiple-testing safeguards directly onto this optimization summary schema.

## Self-Check: PASSED

- `poetry run pytest tests/model/test_optimize.py -q -k "nested or optimize or walk_forward or summary"` -> 17 passed
- `poetry run pytest tests/model/test_optimize.py -q -k "provenance or deterministic or summary"` -> 5 passed, 12 deselected
- `poetry run pytest tests/test_cli.py -q -k "model and optimize and nested"` -> 1 passed, 21 deselected

---
*Phase: 16-promotion-guardrails-and-optimization-safety*
*Completed: 2026-02-26*
