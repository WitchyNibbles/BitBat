---
phase: 25-critical-correctness-remediation
plan: 01
subsystem: autonomous
tags: [retrainer, cli-contract, cv-metrics, subprocess, correctness]

# Dependency graph
requires:
  - phase: 24-audit-baseline
    provides: "AUDIT-REPORT.md identifying CORR-01 and CORR-02 bugs"
provides:
  - "Fixed retrainer subprocess command (no invalid --tau to features build)"
  - "Consistent CV metric key naming (mean_directional_accuracy everywhere)"
  - "CLI contract test catching future argument mismatches"
  - "Round-trip consistency test for cv_summary.json write/read"
affects: [autonomous-retraining, model-cv, monitoring]

# Tech tracking
tech-stack:
  added: []
  patterns: [cli-contract-testing, metric-key-consistency, round-trip-testing]

key-files:
  created:
    - tests/autonomous/test_retrainer_cli_contract.py
    - tests/model/test_cv_metric_roundtrip.py
  modified:
    - src/bitbat/autonomous/retrainer.py
    - src/bitbat/cli.py

key-decisions:
  - "Writer key renamed from average_balanced_accuracy to mean_directional_accuracy; reader cascade kept for backward compat with old cv_summary.json files"
  - "CLI contract test uses Click CliRunner --help parsing to extract valid options dynamically"

patterns-established:
  - "CLI contract testing: capture subprocess args via monkeypatch, validate against real CLI --help output"
  - "Metric round-trip testing: write structure then read via production code path to verify consistency"

requirements-completed: [CORR-01, CORR-02]

# Metrics
duration: 5min
completed: 2026-03-04
---

# Phase 25 Plan 01: Retrainer CLI Contract and CV Metric Key Fix Summary

**Removed invalid --tau from retrainer subprocess command, aligned cv_summary.json key naming to mean_directional_accuracy, added 4 contract/round-trip tests**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-04T19:09:00Z
- **Completed:** 2026-03-04T19:14:28Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Removed `--tau` argument from the `features build` subprocess call in `AutoRetrainer.retrain()` -- this argument does not exist on the CLI command and would cause retraining to fail
- Renamed the cv_summary.json writer key from `average_balanced_accuracy` to `mean_directional_accuracy` eliminating a confusing alias where the key said "balanced accuracy" but stored directional accuracy
- Updated the retrainer reader to look up `mean_directional_accuracy` as the primary key, keeping the champion_decision and RMSE fallback cascades for backward compatibility
- Added 4 new behavioral tests: 2 CLI contract tests and 2 metric round-trip consistency tests

## Task Commits

Each task was committed atomically:

1. **Task 1: Fix retrainer CLI contract and add contract test (CORR-01)** - `37bba9d` (fix)
2. **Task 2: Fix CV metric key naming and add round-trip test (CORR-02)** - `de9cd38` (fix)

## Files Created/Modified
- `src/bitbat/autonomous/retrainer.py` - Removed --tau from features build command; updated _read_cv_score primary key to mean_directional_accuracy
- `src/bitbat/cli.py` - Renamed cv_summary.json aggregate key from average_balanced_accuracy to mean_directional_accuracy
- `tests/autonomous/test_retrainer_cli_contract.py` - CLI contract tests: no --tau arg, all flags valid against real CLI
- `tests/model/test_cv_metric_roundtrip.py` - Round-trip tests: write-then-read consistency, key name match between writer and reader

## Decisions Made
- Kept the champion_decision cascade fallback in `_read_cv_score()` for backward compatibility with older cv_summary.json files that may not have the top-level key
- Used Click CliRunner to dynamically extract valid CLI options rather than hardcoding them, making the contract test self-updating
- Used source inspection (inspect.getsource) in the key-matching test to verify writer and reader reference the same key string

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed CLI import name**
- **Found during:** Task 1 (contract test creation)
- **Issue:** Plan referenced `from bitbat.cli import cli` but the Click group is named `_cli` in the source
- **Fix:** Changed import to `from bitbat.cli import _cli` and updated the CliRunner invocation
- **Files modified:** tests/autonomous/test_retrainer_cli_contract.py
- **Verification:** Both contract tests pass
- **Committed in:** 37bba9d (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Minor import name correction. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- CORR-01 and CORR-02 both resolved with tests
- Full test suite passes (607 tests, 0 failures)
- Ready for 25-02 (next correctness fixes)

## Self-Check: PASSED

All artifacts verified:
- src/bitbat/autonomous/retrainer.py: FOUND
- src/bitbat/cli.py: FOUND
- tests/autonomous/test_retrainer_cli_contract.py: FOUND
- tests/model/test_cv_metric_roundtrip.py: FOUND
- 25-01-SUMMARY.md: FOUND
- Commit 37bba9d: FOUND
- Commit de9cd38: FOUND

---
*Phase: 25-critical-correctness-remediation*
*Completed: 2026-03-04*
