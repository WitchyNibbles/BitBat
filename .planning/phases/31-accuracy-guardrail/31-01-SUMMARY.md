---
phase: 31-accuracy-guardrail
plan: 01
subsystem: monitoring
tags: [autonomous, guardrail, alerting, accuracy, xgboost, yaml-config]

# Dependency graph
requires:
  - phase: 30-fix-and-reset
    provides: "Fixed 3-class XGBoost model + validator tau wiring — realized hit_rate now meaningful"
provides:
  - "check_accuracy_guardrail() module-level function in autonomous/agent.py"
  - "accuracy_guardrail YAML config section in default.yaml"
  - "accuracy_alert_fired key in MonitoringAgent.run_once() result dict"
  - "5 behavioral tests covering all FIXR-04 success criteria"
affects: [31-accuracy-guardrail, 32-tech-debt, 35-xgboost-fix]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Module-level standalone function extracted from MonitoringAgent to enable unit testing without model artifacts on disk"
    - "Mock patch target matches import-from location (bitbat.autonomous.agent.send_alert not bitbat.autonomous.alerting.send_alert)"
    - "Guardrail config mirrors drift_detection pattern: get_runtime_config() fallback with load_config()"

key-files:
  created:
    - tests/autonomous/test_accuracy_guardrail.py
  modified:
    - src/bitbat/autonomous/agent.py
    - src/bitbat/config/default.yaml

key-decisions:
  - "check_accuracy_guardrail() is a module-level function (not a method) so tests can call it without constructing MonitoringAgent (which requires model artifact on disk)"
  - "Mock patch path is bitbat.autonomous.agent.send_alert because agent.py uses from-import; patching the alerting module would have no effect"
  - "Default threshold 0.40 chosen to match FIXR-04 spec; operator can tune via autonomous.accuracy_guardrail.realized_accuracy_threshold"
  - "min_predictions_required=10 prevents warmup false positives during early post-reset period"

patterns-established:
  - "Guardrail functions: standalone module-level, config-injected via parameter, returns bool (alert fired or not)"
  - "Behavioral tests: import standalone function, inject synthetic metrics + config dicts, patch send_alert at agent module level"

requirements-completed: [FIXR-04]

# Metrics
duration: 3min
completed: 2026-03-08
---

# Phase 31 Plan 01: Accuracy Guardrail Summary

**Realized-accuracy WARNING alert via standalone check_accuracy_guardrail() function wired into MonitoringAgent.run_once(), configurable at autonomous.accuracy_guardrail.realized_accuracy_threshold**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-08T18:17:26Z
- **Completed:** 2026-03-08T18:20:55Z
- **Tasks:** 2 (TDD: RED then GREEN)
- **Files modified:** 3

## Accomplishments

- Added `check_accuracy_guardrail()` as a module-level function in `agent.py` — callable in tests without a model file
- Added `accuracy_guardrail` YAML config section with `realized_accuracy_threshold: 0.40`, `min_predictions_required: 10`, `enabled: true`
- Wired guardrail into `MonitoringAgent.run_once()` — result dict now includes `accuracy_alert_fired` key
- 5 behavioral tests pass covering: fires on low accuracy, respects custom threshold, skips insufficient samples, correct alert detail keys, config key present in default.yaml

## Task Commits

Each task was committed atomically:

1. **Task 1: Write failing tests (RED)** - `1f2ee59` (test)
2. **Task 2: Implement guardrail — config + function + agent wiring (GREEN)** - `dab6868` (feat)

_Note: TDD plan — test commit first (RED), then implementation commit (GREEN)_

## Files Created/Modified

- `tests/autonomous/test_accuracy_guardrail.py` - 5 behavioral tests for FIXR-04 success criteria
- `src/bitbat/autonomous/agent.py` - check_accuracy_guardrail() function + run_once() wiring
- `src/bitbat/config/default.yaml` - accuracy_guardrail sub-section under autonomous:

## Decisions Made

- `check_accuracy_guardrail()` is module-level (not a method) so tests can call it without constructing MonitoringAgent (which requires model artifact at `models/{freq}_{horizon}/xgb.json`)
- Mock patch target is `bitbat.autonomous.agent.send_alert` — `agent.py` uses `from bitbat.autonomous.alerting import send_alert`, so patching the alerting module directly has no effect on already-imported references
- `min_predictions_required=10` chosen to match plan spec; prevents warmup false positives in the window immediately after an operator reset

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed mock patch target in tests**
- **Found during:** Task 2 (GREEN verification run)
- **Issue:** Tests used `patch("bitbat.autonomous.alerting.send_alert")` but `agent.py` does `from ... import send_alert`, so the mock didn't intercept calls — test_guardrail_fires_on_low_accuracy failed with `mock.call_count == 0`
- **Fix:** Changed all 4 patch targets to `"bitbat.autonomous.agent.send_alert"` (where the name is bound)
- **Files modified:** `tests/autonomous/test_accuracy_guardrail.py`
- **Verification:** All 5 tests pass after fix
- **Committed in:** `dab6868` (Task 2 commit)

**2. [Rule 1 - Bug] Fixed E501 lint violations in test docstrings**
- **Found during:** Task 2 (lint gate)
- **Issue:** Two docstrings exceeded 100-char line limit — `ruff check` exited 1
- **Fix:** Wrapped long docstrings into multi-line format
- **Files modified:** `tests/autonomous/test_accuracy_guardrail.py`
- **Verification:** `ruff check src/ tests/` exits 0
- **Committed in:** `dab6868` (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (both Rule 1 — bugs found during verification)
**Impact on plan:** Both fixes were necessary for correctness. No scope creep.

## Issues Encountered

- Pre-existing diagnosis test failures in `tests/diagnosis/test_pipeline_stage_trace.py` (3 tests) — require operator `bitbat system reset --yes` + retrain, documented since Phase 30. Not caused by this plan.

## User Setup Required

None — no external service configuration required. The guardrail activates automatically once the operator completes the system reset + retrain cycle (documented in Phase 30 blockers).

## Next Phase Readiness

- FIXR-04 complete — accuracy guardrail fires within one monitor cycle when hit_rate < 0.40
- Phase 31 Plan 01 done; Phase 31 complete (single-plan phase)
- Phase 32 (tech debt) is unblocked and independent of the accuracy recovery track
- Operator still needs to run `bitbat system reset --yes` and retrain to clear pre-fix autonomous.db predictions before the guardrail fires on live data

---
*Phase: 31-accuracy-guardrail*
*Completed: 2026-03-08*
