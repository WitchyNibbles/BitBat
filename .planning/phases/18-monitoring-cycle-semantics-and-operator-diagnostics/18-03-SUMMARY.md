---
phase: 18-monitoring-cycle-semantics-and-operator-diagnostics
plan: "03"
subsystem: monitor-root-cause-diagnostics
tags: [diagnostic-reasons, monitor-run-once, heartbeat, operator-visibility]
requires:
  - phase: 18-monitoring-cycle-semantics-and-operator-diagnostics
    provides: Cycle-state semantics and lifecycle count contracts from Plans 01-02
provides:
  - Stable predictor diagnostic fields (`diagnostic_reason`, `diagnostic_message`) for no-prediction paths
  - Concise cycle-level root-cause diagnostic lines in agent logs and run-once CLI output
  - Heartbeat payload propagation of latest cycle diagnostic state and reason
affects: [autonomous-predictor, monitoring-agent, monitor-cli, monitoring-daemon, operator-diagnostics, v1.3-phase18]
tech-stack:
  added: []
  patterns:
    - Root-cause diagnostics should be represented as a single concise `cycle_diagnostic` line
    - Heartbeat payloads should carry latest cycle diagnostic context when available
key-files:
  created: []
  modified:
    - src/bitbat/autonomous/predictor.py
    - src/bitbat/autonomous/agent.py
    - src/bitbat/cli.py
    - scripts/run_monitoring_agent.py
    - tests/autonomous/test_agent_integration.py
    - tests/test_cli.py
    - tests/test_run_monitoring_agent.py
key-decisions:
  - Standardized predictor diagnostic aliases so downstream surfaces consume one stable schema.
  - Added `cycle_diagnostic` string to monitor cycle payloads and operator-facing run-once output.
  - Extended heartbeat schema with optional cycle diagnostic fields instead of replacing existing metadata.
patterns-established:
  - Operator root-cause visibility should be available without traceback inspection.
  - Heartbeat updates should include both runtime metadata and latest cycle diagnostic semantics.
requirements-completed: [MON-06, MON-04]
duration: 3 min
completed: 2026-02-26
---

# Phase 18 Plan 03: Root-Cause Diagnostic Propagation Summary

**Missing-model and no-prediction root causes now propagate as concise cycle diagnostics across monitor payloads, CLI output, and heartbeat artifacts.**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-26T14:32:09+01:00
- **Completed:** 2026-02-26T14:34:24+01:00
- **Tasks:** 3
- **Files modified:** 7

## Accomplishments

- Added stable diagnostic aliases on predictor outputs so no-prediction reasons are schema-consistent.
- Introduced concise cycle diagnostic root-cause lines in monitoring agent payload/log output and run-once CLI output.
- Propagated cycle diagnostic state/reason fields into heartbeat payload updates for daemon operators.

## Task Commits

Each task was committed atomically:

1. **Task 1: Standardize predictor no-prediction diagnostic reason schema** - `33613f9` (feat)
2. **Task 2: Surface concise cycle diagnostic root-cause lines in agent/CLI output** - `dc6b545` (feat)
3. **Task 3: Propagate cycle diagnostics into heartbeat updates and lock with tests** - `c1a3c4e` (feat)

## Files Created/Modified

- `src/bitbat/autonomous/predictor.py` - predictor result payload now always includes `diagnostic_reason`/`diagnostic_message` aliases.
- `src/bitbat/autonomous/agent.py` - cycle payload/logging now emits a single `cycle_diagnostic` root-cause summary line.
- `src/bitbat/cli.py` - run-once output now includes a concise cycle diagnostic line.
- `scripts/run_monitoring_agent.py` - heartbeat updates now carry cycle diagnostic state/reason fields when present.
- `tests/autonomous/test_agent_integration.py` - regression test for stable predictor diagnostic field schema.
- `tests/test_cli.py` - regression test for cycle root-cause diagnostic output line.
- `tests/test_run_monitoring_agent.py` - regression test for heartbeat diagnostic field propagation.

## Decisions Made

- Retained original `reason`/`message` fields while adding diagnostic aliases to preserve compatibility.
- Used a derived single-line `cycle_diagnostic` for operator readability.
- Kept heartbeat diagnostic fields optional to avoid breaking startup/error heartbeat writes.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None.

## Next Phase Readiness

- Phase 18 now has explicit cycle semantics, status lifecycle counts, and root-cause diagnostics across run-once/status/heartbeat surfaces.
- Phase verification can validate MON-04/05/06 as complete and route to Phase 19 quality gates.

## Self-Check: PASSED

- `poetry run pytest tests/autonomous/test_agent_integration.py -q -k "missing_model or diagnostic or no_prediction"` -> 3 passed, 9 deselected
- `poetry run pytest tests/test_cli.py -q -k "monitor and run_once and (reason or root or cause or missing_model or no_prediction)"` -> 2 passed, 30 deselected
- `poetry run pytest tests/test_run_monitoring_agent.py -q -k "diagnostic or heartbeat or error"` -> 3 passed

---
*Phase: 18-monitoring-cycle-semantics-and-operator-diagnostics*
*Completed: 2026-02-26*
