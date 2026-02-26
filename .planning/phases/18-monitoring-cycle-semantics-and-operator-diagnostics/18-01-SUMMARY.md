---
phase: 18-monitoring-cycle-semantics-and-operator-diagnostics
plan: "01"
subsystem: monitor-cycle-semantics
tags: [monitoring-agent, predictor-diagnostics, cycle-state, monitor-cli]
requires:
  - phase: 17-runtime-pair-alignment-and-startup-guardrails
    provides: Runtime pair startup guardrails and schema-safe monitor execution baseline
provides:
  - Structured no-prediction status/reason contract in live predictor output
  - Explicit cycle-state payload fields for prediction and realization semantics
  - `bitbat monitor run-once` output lines that mirror cycle-state semantics
affects: [autonomous-predictor, monitoring-agent, monitor-cli, operator-diagnostics, v1.3-phase18]
tech-stack:
  added: []
  patterns:
    - Predictor outputs always include `status`, `reason`, and `message` for no-prediction flows
    - Monitoring cycle payload includes explicit `prediction_state` and `realization_state` semantics
key-files:
  created: []
  modified:
    - src/bitbat/autonomous/predictor.py
    - src/bitbat/autonomous/agent.py
    - src/bitbat/cli.py
    - tests/autonomous/test_agent_integration.py
    - tests/test_cli.py
key-decisions:
  - Standardized predictor no-prediction outcomes on stable reason codes (`missing_model`, `insufficient_data`, etc.).
  - Encoded cycle semantics as explicit fields (`prediction_state`, `prediction_reason`, `realization_state`) instead of inferring from null payloads.
  - Printed cycle semantics directly in run-once CLI output for one-command operator visibility.
patterns-established:
  - No-prediction monitor paths now carry machine-readable reason codes and human-readable messages.
  - Operator-facing monitor outputs should mirror underlying cycle-state payload fields without inference.
requirements-completed: [MON-04]
duration: 4 min
completed: 2026-02-26
---

# Phase 18 Plan 01: Monitoring Cycle Semantics and Operator Diagnostics Summary

**Monitor cycles now expose explicit prediction-generation and realization semantics with stable no-prediction reasons end-to-end.**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-26T14:22:04+01:00
- **Completed:** 2026-02-26T14:24:32+01:00
- **Tasks:** 3
- **Files modified:** 5

## Accomplishments

- Added structured predictor result contracts so no-prediction paths provide stable `status`/`reason`/`message` diagnostics.
- Extended `MonitoringAgent.run_once` payloads with explicit prediction and realization state semantics.
- Updated `bitbat monitor run-once` output to surface cycle-state and reason lines for operators.

## Task Commits

Each task was committed atomically:

1. **Task 1: Define explicit no-prediction status contract in live predictor** - `ece0ea2` (feat)
2. **Task 2: Build cycle-state payload contract in monitoring agent** - `7063606` (feat)
3. **Task 3: Surface cycle semantics through `bitbat monitor run-once` output** - `fa08154` (feat)

## Files Created/Modified

- `src/bitbat/autonomous/predictor.py` - predictor now returns structured diagnostics for all no-prediction conditions.
- `src/bitbat/autonomous/agent.py` - cycle payload includes `prediction_state`, `prediction_reason`, and `realization_state`.
- `src/bitbat/cli.py` - run-once CLI prints cycle-state semantic lines.
- `tests/autonomous/test_agent_integration.py` - regression tests for predictor reason codes and cycle-state payload semantics.
- `tests/test_cli.py` - regression test for run-once cycle semantic output.

## Decisions Made

- Predictor status contract was normalized to avoid ambiguous `None` payloads in no-data cases.
- Realization state was defined with explicit enum-like states: `none`, `pending`, `realized`.
- CLI output was aligned to payload semantics so operators see the same state contract without reading logs.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None.

## Next Phase Readiness

- Plan 18-02 can now build on stable cycle semantics by adding pair-scoped lifecycle counts to monitor status.
- Plan 18-03 can propagate the same diagnostic reason contract into heartbeat and concise root-cause log lines.

## Self-Check: PASSED

- `poetry run pytest tests/autonomous/test_agent_integration.py -q -k "predict or no_prediction or status or reason"` -> 2 passed, 7 deselected
- `poetry run pytest tests/autonomous/test_agent_integration.py -q -k "cycle_state or realization or metrics"` -> 2 passed, 9 deselected
- `poetry run pytest tests/test_cli.py -q -k "monitor and run_once and (state or semantics or pending or realized)"` -> 1 passed, 28 deselected

---
*Phase: 18-monitoring-cycle-semantics-and-operator-diagnostics*
*Completed: 2026-02-26*
