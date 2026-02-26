---
phase: 19-regression-gates-and-runbook-hardening
plan: "02"
subsystem: docs
tags: [monitor-runbook, operations, systemd, config-wiring, docs-contract-tests]
requires:
  - phase: 19-regression-gates-and-runbook-hardening
    provides: Phase 19 Plan 01 release-gate regression coverage
provides:
  - Dedicated monitor operations runbook with supported config wiring and diagnostics workflow
  - Deployment service template aligned to explicit `BITBAT_CONFIG` and `--config` contract
  - Automated documentation contract tests preventing runbook/service guidance drift
affects: [operator-docs, deployment, release-readiness, v1.3-phase19]
tech-stack:
  added: []
  patterns:
    - Operations docs are treated as contract surfaces and regression-tested
    - Service templates should mirror operator runbook startup wiring exactly
key-files:
  created:
    - docs/monitor-operations-runbook.md
    - tests/test_monitor_runbook_contract.py
  modified:
    - docs/README.md
    - docs/usage-guide.md
    - docs/monitoring_strategy.md
    - docs/testing-quality.md
    - deployment/bitbat-monitor.service
key-decisions:
  - Centralized monitor startup, diagnostics, and remediation guidance into one runbook to avoid split-brain docs.
  - Enforced service template startup via `BITBAT_CONFIG` plus `--config` passthrough for explicit pair resolution.
  - Added docs contract tests so runbook/service guidance drift fails fast in CI.
patterns-established:
  - Operator documentation for critical runtime safety paths should be test-locked.
  - `make test-release` remains the canonical pre-release verification command.
requirements-completed: [QUAL-07, QUAL-08, QUAL-09]
duration: 1 min
completed: 2026-02-26
---

# Phase 19 Plan 02: Regression Gates and Runbook Hardening Summary

**Monitor operations now have a single runbook with explicit config wiring, diagnostic interpretation, and schema remediation that is enforced by documentation contract tests.**

## Performance

- **Duration:** 1 min
- **Started:** 2026-02-26T16:57:40+01:00
- **Completed:** 2026-02-26T16:58:13+01:00
- **Tasks:** 3
- **Files modified:** 8

## Accomplishments

- Published a dedicated monitor operations runbook covering `--config`/`BITBAT_CONFIG`, startup guardrails, cycle diagnostics, status interpretation, schema remediation, and release checks.
- Aligned deployment service template and monitoring strategy docs to the same explicit startup wiring contract.
- Added docs contract tests and testing-quality references to prevent runbook/service guidance drift.

## Task Commits

Each task was committed atomically:

1. **Task 1: Publish monitor operations runbook with explicit config-wiring and diagnostic workflow** - `69eaa42` (docs)
2. **Task 2: Align deployment monitor service template with documented config contract** - `96f64fd` (docs)
3. **Task 3: Add automated documentation contract checks for monitor runbook wiring** - `9020e5e` (test)

## Files Created/Modified

- `docs/monitor-operations-runbook.md` - Canonical monitor operations runbook for startup wiring, diagnostics, remediation, and release verification.
- `docs/README.md` - Added runbook to documentation hub navigation.
- `docs/usage-guide.md` - Added monitor operations section linking runbook and release command.
- `docs/monitoring_strategy.md` - Updated execution model to explicit config wiring and runbook reference.
- `deployment/bitbat-monitor.service` - Added explicit `BITBAT_CONFIG` environment and `--config` startup passthrough.
- `tests/test_monitor_runbook_contract.py` - Added contract tests for runbook/service/doc anchor integrity.
- `docs/testing-quality.md` - Registered docs contract test coverage and release regression command.
- `tests/api/test_metrics.py` - Normalized incompatible-schema fixture to keep release-gate schema metrics expectations deterministic.

## Decisions Made

- Standardized on a single monitor operations runbook instead of scattering guidance across multiple docs.
- Made service config wiring explicit in both environment and startup command.
- Used string-contract tests for docs stability to keep regression checks lightweight and deterministic.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Release gate failed due overly degraded incompatible-schema fixture**
- **Found during:** Phase-level verification (`make test-release`)
- **Issue:** `tests/api/test_metrics.py` built an incompatible DB with only one recreated table, which now produced a large missing-column count and broke intended auto-upgrade assertions.
- **Fix:** Initialized the runtime schema first, then downgraded only `prediction_outcomes` to retain the targeted one-column compatibility scenario.
- **Files modified:** `tests/api/test_metrics.py`
- **Verification:** `poetry run pytest tests/api/test_metrics.py::TestMetricsWithIncompatibleSchema::test_schema_reports_incompatible -q`; `make test-release`
- **Committed in:** `d81952d`

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Preserved deterministic release-gate behavior and kept schema-compatibility assertions aligned with intended fixture scope.

## Issues Encountered

None.

## User Setup Required

None.

## Next Phase Readiness

- Phase 19 now has both regression gate hardening and operator runbook contract hardening completed.
- Phase-level verification can run and close milestone v1.3 if no gaps are found.

## Self-Check: PASSED

- `rg -n "BITBAT_CONFIG|--config|monitor run-once|monitor status|cycle diagnostic|make test-release" docs/monitor-operations-runbook.md docs/usage-guide.md docs/README.md` -> required anchors present
- `rg -n "BITBAT_CONFIG|ExecStart|run_monitoring_agent.py|--config" deployment/bitbat-monitor.service docs/monitor-operations-runbook.md docs/monitoring_strategy.md` -> service/docs wiring aligned
- `poetry run pytest tests/test_monitor_runbook_contract.py -q` -> 3 passed
- `make test-release` -> 135 passed, 43 deselected

---
*Phase: 19-regression-gates-and-runbook-hardening*
*Completed: 2026-02-26*
