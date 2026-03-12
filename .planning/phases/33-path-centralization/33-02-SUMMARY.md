---
phase: 33-path-centralization
plan: 02
subsystem: config
tags: [config, pathlib, cli, api, autonomous, path-centralization]

requires:
  - phase: 33-path-centralization-01
    provides: Path helpers and structural test scaffold for DEBT-02
provides:
  - zero remaining Path("models") or Path("metrics") literals in src/
  - cli, api, model, autonomous, and backtest modules routed through config path helpers
  - structural path-centralization tests turned green
affects: [34-db-unification, config, cli, api, autonomous]

tech-stack:
  added: []
  patterns:
    - artifact path consumers call resolve_models_dir() or resolve_metrics_dir() instead of constructing literals

key-files:
  created: []
  modified:
    - src/bitbat/model/evaluate.py
    - src/bitbat/model/train.py
    - src/bitbat/autonomous/predictor.py
    - src/bitbat/autonomous/continuous_trainer.py
    - src/bitbat/autonomous/agent.py
    - src/bitbat/autonomous/retrainer.py
    - src/bitbat/backtest/metrics.py
    - src/bitbat/cli/commands/monitor.py
    - src/bitbat/cli/commands/system.py
    - src/bitbat/cli/commands/model.py
    - src/bitbat/api/routes/health.py
    - src/bitbat/api/routes/analytics.py

key-decisions:
  - "Functions with metrics path defaults now resolve lazily from config using None sentinels to avoid definition-time path capture"
  - "Classes that already hold config-derived state resolve model directories once during initialization and reuse them for artifact access"

patterns-established:
  - "Hardcoded artifact directories are forbidden in src/; structural grep tests enforce the helper-only path contract"

requirements-completed: [DEBT-02]

duration: 7min
completed: 2026-03-12
---

# Phase 33 Plan 02: Path Centralization Summary

**All remaining model and metrics artifact paths now resolve through the config loader, and the hardcoded-path structural gate is fully green**

## Performance

- **Duration:** 7 min
- **Started:** 2026-03-12T15:02:00Z
- **Completed:** 2026-03-12T15:08:31Z
- **Tasks:** 2
- **Files modified:** 12

## Accomplishments

- Replaced all remaining `Path("models")` and `Path("metrics")` usages across model, autonomous, backtest, CLI, and API modules
- Converted metrics-writing functions in [evaluate.py](/home/eimi/projects/ai-btc-predictor/src/bitbat/model/evaluate.py) to lazy config resolution so defaults no longer bake in hardcoded directories
- Turned [test_path_resolution.py](/home/eimi/projects/ai-btc-predictor/tests/config/test_path_resolution.py) fully green and verified the broader suite stays green when the known diagnosis blocker is excluded

## Task Commits

Each task was committed atomically:

1. **Task 1: Replace Path("models") / Path("metrics") in model/, autonomous/, backtest/** - `8661992` (feat)
2. **Task 2: Replace Path("models") / Path("metrics") in cli/commands/ and api/routes/** - `08d33c1` (feat)

## Files Created/Modified

- `src/bitbat/model/evaluate.py` - lazy metrics directory resolution for diagnostics and regression artifact writes
- `src/bitbat/model/train.py` - default model artifact path now uses `resolve_models_dir()`
- `src/bitbat/autonomous/predictor.py` - predictor model directory now follows config
- `src/bitbat/autonomous/continuous_trainer.py` - retraining diagnostics/model deployment paths now follow config
- `src/bitbat/autonomous/agent.py` - monitor preflight checks now use config-resolved model path
- `src/bitbat/autonomous/retrainer.py` - CV summary path now uses config-resolved metrics directory
- `src/bitbat/backtest/metrics.py` - backtest summary artifacts now write through `resolve_metrics_dir()`
- `src/bitbat/cli/commands/monitor.py` - monitoring snapshot output now respects config-resolved metrics directory
- `src/bitbat/cli/commands/system.py` - system reset now deletes the configured models directory
- `src/bitbat/cli/commands/model.py` - CV and optimization summaries now respect config-resolved metrics directory
- `src/bitbat/api/routes/health.py` - model health probe now uses config-resolved model path
- `src/bitbat/api/routes/analytics.py` - analytics endpoints now use config-resolved model path

## Decisions Made

- Kept directory defaults relative so existing cwd-based workflows continue to function unless the operator overrides config
- Resolved paths inside functions or constructors, not at module import time, to avoid stale config capture

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Removed stale Path imports after the sweep**
- **Found during:** Task 2 verification
- **Issue:** `ruff check` flagged unused `Path` imports left behind in monitor/agent modules after replacing the literals
- **Fix:** Removed the unused imports and re-ran lint
- **Files modified:** `src/bitbat/autonomous/agent.py`, `src/bitbat/cli/commands/monitor.py`
- **Verification:** `poetry run ruff check src/bitbat/model/ src/bitbat/autonomous/ src/bitbat/backtest/ src/bitbat/cli/commands/ src/bitbat/api/routes/`
- **Committed in:** `08d33c1`

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Minor lint cleanup only. No scope change.

## Issues Encountered

- The repo still has a known runtime-data-dependent diagnosis failure in `tests/diagnosis/test_pipeline_stage_trace.py::test_serving_direction_is_balanced` unless the operator resets and retrains. Verification for this plan therefore used the full suite with that single blocker excluded.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 33 is functionally complete: no hardcoded `models` or `metrics` path literals remain in `src/`
- `tests/config/test_path_resolution.py` is fully green
- `poetry run pytest tests/ -x --ignore=tests/diagnosis/test_pipeline_stage_trace.py` passed with 660 tests green
- Phase 34 (DB Unification) is unblocked

---
*Phase: 33-path-centralization*
*Completed: 2026-03-12*
