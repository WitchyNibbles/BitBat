---
phase: 32-cli-decomposition
plan: 03
subsystem: cli
tags: [click, xgboost, cli-decomposition, c901, monkeypatch]

requires:
  - phase: 32-cli-decomposition-01
    provides: cli/ package skeleton, _helpers.py with 21 private helpers
  - phase: 32-cli-decomposition-02
    provides: 8 command group modules in commands/; __init__.py reduced

provides:
  - commands/model.py with model group + 4 commands (cv, optimize, train, infer)
  - commands/monitor.py with monitor group + 5 commands (refresh, run-once, start, status, snapshots)
  - cli/__init__.py as thin registration-only file (83 lines, no command bodies)
  - model_cv C901 complexity refactored into 5 private helper functions
  - All 15 monkeypatch targets in tests/test_cli.py updated to commands.* paths
  - inspect.getsource target in test_cv_metric_roundtrip.py updated
  - DEBT-01 fully satisfied: zero noqa:C901 suppressions in cli/

affects:
  - future CLI changes
  - test_cli.py monkeypatch patterns
  - any code inspecting bitbat.cli module source

tech-stack:
  added: []
  patterns:
    - "CLI decomposition: model_cv complexity extracted into 5 private _resolve_*/run_*/build_* helpers"
    - "Re-exports in __init__.py with noqa:F401 for test monkeypatch compatibility"
    - "Lazy imports in command bodies (from bitbat.autonomous... inside functions) to avoid circular deps"

key-files:
  created:
    - src/bitbat/cli/commands/model.py
    - src/bitbat/cli/commands/monitor.py
  modified:
    - src/bitbat/cli/__init__.py
    - tests/test_cli.py
    - tests/model/test_cv_metric_roundtrip.py
    - tests/dataset/test_public_api.py

key-decisions:
  - "model_cv refactored via 5 private helpers (_resolve_cv_embargo_purge, _resolve_cv_window_spec, _run_cv_folds, _build_family_metrics, _run_champion_selection) to bring C901 below 10 without suppressions"
  - "test_public_api.py updated to point to cli/__init__.py instead of cli.py (monolith deleted in Phase 32-01)"
  - "Pre-existing test failures (test_cli_features_build_*, tests/diagnosis/*) excluded — confirmed failing before Phase 32-03 changes"
  - "Re-exports in __init__.py kept for documentation; tests must use commands.* monkeypatch paths"

patterns-established:
  - "CLI C901 complexity reduction: extract 5 domain helpers, reduce command body to orchestration-only ~50 lines"

requirements-completed:
  - DEBT-01

duration: 25min
completed: 2026-03-12
---

# Phase 32 Plan 03: CLI Decomposition — Model and Monitor Summary

**model_cv C901 refactored into 5 private helpers, all 10 command groups in dedicated files, cli/__init__.py is 83-line thin registration layer with zero noqa:C901 suppressions**

## Performance

- **Duration:** ~25 min
- **Started:** 2026-03-12T10:25:00Z
- **Completed:** 2026-03-12T10:38:00Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments

- Created `commands/model.py` (774 lines) with model_cv refactored: 5 extracted private helpers reduce cyclomatic complexity below C901 threshold, zero suppressions
- Created `commands/monitor.py` with all 5 monitor commands verbatim-migrated from monolith
- Rewrote `cli/__init__.py` to 83-line thin file: imports, `_cli` group, 10 `add_command()` calls, re-exports, `main()`
- Updated all 15 monkeypatch targets in `tests/test_cli.py` to point at `bitbat.cli.commands.*` module paths
- Updated `inspect.getsource` target in `test_cv_metric_roundtrip.py` to `bitbat.cli.commands.model`
- Fixed `test_public_api.py` to use `cli/__init__.py` path (old `cli.py` deleted in Plan 01)
- `ruff C901` passes with zero violations on entire `src/bitbat/cli/`
- `poetry run lint-imports` passes; `bitbat --help` shows all 10 command groups

## Task Commits

1. **Task 1: Create commands/model.py with model_cv C901 refactor** - `b97c792` (feat)
2. **Task 2: Create commands/monitor.py, wire all 10 groups, update tests** - `dc1237d` (feat)

**Plan metadata:** (this commit)

## Files Created/Modified

- `src/bitbat/cli/commands/model.py` - model group + 4 commands + 5 C901 helper functions
- `src/bitbat/cli/commands/monitor.py` - monitor group + 5 commands
- `src/bitbat/cli/__init__.py` - thin registration file (83 lines, no command bodies)
- `tests/test_cli.py` - 15 monkeypatch targets updated to commands.* paths
- `tests/model/test_cv_metric_roundtrip.py` - inspect.getsource target updated
- `tests/dataset/test_public_api.py` - cli.py path updated to cli/__init__.py

## Decisions Made

- model_cv extracted into 5 domain-aligned helpers: `_resolve_cv_embargo_purge`, `_resolve_cv_window_spec`, `_run_cv_folds`, `_build_family_metrics`, `_run_champion_selection`
- `test_public_api.py` path fix is Rule 1 auto-fix: pre-existing test broke due to Plan 32-01 deleting `cli.py`
- Pre-existing failures in `tests/diagnosis/` and `test_cli_features_build_*` excluded — confirmed failing before these changes via `git stash` verification

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Updated test_public_api.py cli.py path reference**
- **Found during:** Task 2 (full test run)
- **Issue:** `test_no_private_imports_in_callers` hardcoded `src/bitbat/cli.py` path; file was deleted in Phase 32-01
- **Fix:** Updated path to `src/bitbat/cli/__init__.py` which is the equivalent entry point
- **Files modified:** tests/dataset/test_public_api.py
- **Verification:** All 6 tests in test_public_api.py pass
- **Committed in:** dc1237d (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - Bug)
**Impact on plan:** Essential structural fix — test was asserting against a deleted file path.

## Issues Encountered

- Unused imports in model.py from the plan's prescribed import list (`_data_path`, `_feature_dataset_path`, `_predictions_path`, `UTC`, `datetime`, `Literal`, `default_model_artifact_path`, `ensure_utc`) — removed and re-linted cleanly.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 32 (CLI Decomposition) is now complete: all 10 command groups in dedicated files, zero noqa:C901 suppressions, `cli/__init__.py` is a thin registration layer
- DEBT-01 fully resolved
- Phases 33-35 (DEBT-02/03/04) are unblocked

## Self-Check: PASSED

All created files exist and all task commits verified in git history.

---
*Phase: 32-cli-decomposition*
*Completed: 2026-03-12*
