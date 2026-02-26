---
phase: 17-runtime-pair-alignment-and-startup-guardrails
plan: "01"
subsystem: monitor-startup-guardrails
tags: [monitor, startup, config-source, runtime-alignment, model-artifact]
requires:
  - phase: 16-promotion-guardrails-and-optimization-safety
    provides: Stable v1.2 baseline and monitor/runtime diagnostics context
provides:
  - Runtime config source/path introspection with explicit monitor startup reporting
  - Startup-blocking model artifact preflight for resolved freq/horizon
  - Regression coverage ensuring startup cannot silently run mismatched runtime/model pairs
affects: [config-loader, monitor-cli, autonomous-agent, monitor-tests, v1.3-phase17]
tech-stack:
  added: []
  patterns:
    - Monitor startup must announce resolved config source/path and freq/horizon before execution
    - Missing runtime model artifact for resolved pair is a hard startup failure with remediation
key-files:
  created: []
  modified:
    - src/bitbat/config/loader.py
    - src/bitbat/cli.py
    - src/bitbat/autonomous/agent.py
    - tests/test_cli.py
    - tests/autonomous/test_agent_integration.py
    - tests/autonomous/test_phase8_d1_monitor_schema_complete.py
    - tests/autonomous/test_session3_complete.py
key-decisions:
  - Added config-source tracking in the loader (`explicit`, `env`, `default`) so startup metadata is deterministic.
  - Enforced model artifact preflight in `MonitoringAgent.__init__` to block silent no-model loops.
  - Surfaced startup preflight failures as operator-facing Click exceptions with clear remediation steps.
patterns-established:
  - Monitor entrypoints (`run-once`, `start`) must emit startup context before attempting DB/agent operations.
  - Monitoring agent integration tests must seed model artifacts for the active runtime pair.
requirements-completed: [ALGN-01, ALGN-02]
duration: 28 min
completed: 2026-02-26
---

# Phase 17 Plan 01: Runtime Alignment and Startup Guardrails Summary

**Monitor startup now reports config provenance and blocks immediately when runtime model artifacts are missing.**

## Performance

- **Duration:** 28 min
- **Completed:** 2026-02-26T12:56:58Z
- **Tasks:** 3
- **Files modified:** 7

## Accomplishments

- Extended config loader runtime metadata to expose active config source and resolved path.
- Added monitor startup context output (`config source/path`, `freq`, `horizon`) before agent startup.
- Added strict startup preflight requiring the resolved runtime model artifact in `MonitoringAgent`.
- Added regression coverage for startup context and fail-fast missing-model behavior.

## Task Commits

Implementation was committed as a single atomic plan commit (coupled file touchpoints across startup flow):

1. **Plan 17-01 implementation** - `c7afadc` (feat)

## Files Created/Modified

- `src/bitbat/config/loader.py` - runtime config source tracking helpers.
- `src/bitbat/cli.py` - startup context logging and missing-model startup exception mapping.
- `src/bitbat/autonomous/agent.py` - model artifact preflight guard.
- `tests/test_cli.py` - startup context + missing-artifact CLI regressions.
- `tests/autonomous/test_agent_integration.py` - startup preflight coverage and runtime fixture updates.
- `tests/autonomous/test_phase8_d1_monitor_schema_complete.py` - runtime fixture updates for preflight.
- `tests/autonomous/test_session3_complete.py` - runtime fixture updates for preflight.

## Decisions Made

- Preflight ownership is in agent initialization so both CLI and daemon startup paths inherit guardrails.
- Startup output uses loader-level source/path data to keep monitor diagnostics aligned with runtime config behavior.

## Deviations from Plan

- Task-level commits were consolidated into one plan-level atomic commit due tightly coupled edits in shared monitor startup code paths.

## Issues Encountered

- Existing integration tests assumed no startup model preflight. Fixed by seeding deterministic model artifact fixtures per runtime pair.

## User Setup Required

None.

## Next Phase Readiness

- Runtime alignment and startup guardrails are in place for heartbeat metadata propagation and schema contract extension.
- Wave 2 heartbeat work can rely on stable config-source semantics introduced here.

## Self-Check: PASSED

- `poetry run pytest tests/test_cli.py -q -k "test_cli_monitor_run_once or test_cli_monitor_start_reports_startup_context or test_cli_monitor_run_once_missing_model_artifact_message or test_cli_monitor_start_missing_model_artifact_message"` -> 7 passed
- `poetry run pytest tests/autonomous/test_agent_integration.py tests/test_cli.py -q -k "monitor and (preflight or artifact or xgb or startup)"` -> 4 passed, 31 deselected
- `poetry run pytest tests/autonomous/test_agent_integration.py tests/test_cli.py -q -k "ALGN or monitor and (startup or config or artifact)"` -> 4 passed, 31 deselected

---
*Phase: 17-runtime-pair-alignment-and-startup-guardrails*
*Completed: 2026-02-26*
