---
phase: 33-path-centralization
plan: 01
subsystem: config
tags: [config, pathlib, tdd, path-centralization]

requires:
  - phase: 32-cli-decomposition
    provides: Stable v1.6 baseline before DEBT-02 path work
provides:
  - resolve_models_dir() and resolve_metrics_dir() in config loader
  - default.yaml models_dir and metrics_dir keys
  - test scaffold that gates the later hardcoded path sweep
affects: [33-02, config, models, metrics]

tech-stack:
  added: []
  patterns:
    - config-driven artifact directory resolution with cwd-relative defaults

key-files:
  created:
    - tests/config/test_path_resolution.py
  modified:
    - src/bitbat/config/loader.py
    - src/bitbat/config/default.yaml

key-decisions:
  - "models_dir and metrics_dir default to relative strings so monkeypatch.chdir-based tests remain compatible"
  - "Structural grep tests live in the same scaffold file and intentionally stay red until Plan 33-02 sweeps all remaining call sites"

patterns-established:
  - "Artifact path helpers must resolve lazily from runtime config, not at module import time"

requirements-completed: [DEBT-02]

duration: 3min
completed: 2026-03-12
---

# Phase 33 Plan 01: Path Centralization Summary

**Config loader now resolves models and metrics directories from YAML, and a new test scaffold gates the remaining hardcoded path sweep**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-12T14:58:36Z
- **Completed:** 2026-03-12T15:01:34Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- Added `resolve_models_dir()` and `resolve_metrics_dir()` to [loader.py](/home/eimi/projects/ai-btc-predictor/src/bitbat/config/loader.py)
- Added `models_dir` and `metrics_dir` defaults to [default.yaml](/home/eimi/projects/ai-btc-predictor/src/bitbat/config/default.yaml)
- Created [test_path_resolution.py](/home/eimi/projects/ai-btc-predictor/tests/config/test_path_resolution.py) with four helper-behavior tests plus two structural grep gates

## Task Commits

Each task was committed atomically:

1. **Task 1: Write failing tests for resolve_models_dir and resolve_metrics_dir** - `654e1dc` (test)
2. **Task 2: Implement resolve_models_dir and resolve_metrics_dir** - `f0416b6` (feat)

## Files Created/Modified

- `tests/config/test_path_resolution.py` - helper behavior tests and structural grep gates for hardcoded `Path("models")` and `Path("metrics")`
- `src/bitbat/config/loader.py` - canonical models/metrics directory resolvers
- `src/bitbat/config/default.yaml` - config keys for models and metrics paths

## Decisions Made

- Kept helper fallbacks cwd-relative instead of absolute to preserve existing test/runtime behavior unless config overrides them
- Used direct imports from `bitbat.config.loader` in the new test file so the RED phase fails immediately when helpers are missing

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- `poetry run pytest tests/ -x --ignore=tests/config/test_path_resolution.py` hit the known pre-existing diagnosis failure in `tests/diagnosis/test_pipeline_stage_trace.py::test_serving_direction_is_balanced` because the local runtime data has not been reset/retrained. This did not affect the new helper behavior tests.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plan 33-01 is complete: the four helper behavior tests are green
- The two structural grep tests remain intentionally red until Plan 33-02 replaces the remaining hardcoded call sites
- Phase 33 can proceed directly to the source sweep in `33-02-PLAN.md`

---
*Phase: 33-path-centralization*
*Completed: 2026-03-12*
