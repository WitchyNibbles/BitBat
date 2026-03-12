---
phase: 32-cli-decomposition
plan: 01
subsystem: cli
tags: [click, python, refactoring, package-conversion]

# Dependency graph
requires:
  - phase: 31-accuracy-guardrail
    provides: "Stable codebase with passing test suite as baseline"
provides:
  - "src/bitbat/cli/ package replacing flat src/bitbat/cli.py"
  - "src/bitbat/cli/_helpers.py with 21 extracted private helper functions"
  - "src/bitbat/cli/commands/__init__.py empty barrel file"
  - "All bitbat.cli.* monkeypatch targets remain intact at original import paths"
affects: [32-02, 32-03]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "cli-package-pattern: CLI module converted from flat file to package; helpers extracted to _helpers.py; main namespace preserved for monkeypatch compatibility"

key-files:
  created:
    - src/bitbat/cli/__init__.py
    - src/bitbat/cli/_helpers.py
    - src/bitbat/cli/commands/__init__.py
  modified:
    - src/bitbat/cli.py (deleted — replaced by package)

key-decisions:
  - "21 private _* helpers moved verbatim to _helpers.py; no behavioral changes"
  - "noqa: F401 on all re-exported domain symbols in __init__.py — required for monkeypatch compatibility where tests patch bitbat.cli.xgb, bitbat.cli.walk_forward, etc."
  - "_feature_dataset_path and _monitor_config_source_label not re-imported into __init__.py (only used inside _helpers.py internally)"
  - "cli.py deleted only after cli/__init__.py fully written — Python will not coexist both at same import path"

patterns-established:
  - "Import surface preservation: all 17+ monkeypatched symbols kept in bitbat.cli namespace via noqa: F401 re-exports"
  - "Helper extraction pattern: private helpers in _helpers.py, re-imported by __init__.py for use in command bodies"

requirements-completed: [DEBT-01]

# Metrics
duration: 20min
completed: 2026-03-12
---

# Phase 32 Plan 01: CLI Package Skeleton Summary

**1817-line cli.py converted to cli/ package: 21 private helpers extracted to _helpers.py, all monkeypatch targets preserved, 41 tests pass**

## Performance

- **Duration:** ~20 min
- **Started:** 2026-03-12T08:55:09Z
- **Completed:** 2026-03-12T09:15:00Z
- **Tasks:** 3
- **Files modified:** 4 (3 created, 1 deleted)

## Accomplishments
- Converted `src/bitbat/cli.py` flat file to `src/bitbat/cli/` package without any behavioral change
- Extracted all 21 `def _*` private helper functions into `src/bitbat/cli/_helpers.py`
- Created `src/bitbat/cli/commands/__init__.py` empty barrel for Plans 02/03
- All 17+ monkeypatched symbols still importable from `bitbat.cli` namespace
- `poetry run lint-imports` passes (1 contract kept, 0 broken)
- 41 targeted tests pass including test_cli, retrainer_cli_contract, cv_metric_roundtrip

## Task Commits

Each task was committed atomically:

1. **Task 1: Create _helpers.py — extract all private helper functions** - `e3df7ed` (feat)
2. **Task 2: Convert cli.py to cli/ package — create __init__.py and commands/__init__.py** - `163c0db` (feat)
3. **Task 3: Verify import architecture and full test suite baseline** - `24690ba` (chore)

## Files Created/Modified
- `src/bitbat/cli/__init__.py` - Package init: full CLI (command groups + commands) importing helpers from _helpers.py; re-exports all monkeypatched symbols
- `src/bitbat/cli/_helpers.py` - All 21 shared private helper functions, self-contained
- `src/bitbat/cli/commands/__init__.py` - Empty barrel file marking the commands sub-package
- `src/bitbat/cli.py` - Deleted (replaced by package)

## Decisions Made
- Used `# noqa: F401` on all re-exported domain symbols (xgb, fit_xgb, walk_forward, etc.) since they must remain bound in `bitbat.cli` namespace for monkeypatch compatibility — ruff would otherwise flag them as unused
- Removed `_feature_dataset_path` and `_monitor_config_source_label` from `__init__.py` imports since they are only used internally within `_helpers.py`, not directly in command bodies
- Removed `get_runtime_config`, `get_runtime_config_path`, `get_runtime_config_source` from `__init__.py` imports since these are now accessed via `_helpers._config()` and `_helpers._emit_monitor_startup_context()`

## Deviations from Plan

None - plan executed exactly as written. The lint fixes (removing unused imports, fixing import sort order) were standard ruff auto-fixes required by the CLAUDE.md pre-commit checklist.

## Issues Encountered
- ruff flagged several unused imports in both `_helpers.py` and `__init__.py` — `json`, `Iterable`, `UTC`, `Literal`, `NoReturn`, `get_runtime_config`, `get_runtime_config_path`, `get_runtime_config_source`, `default_model_artifact_path`. All removed by ruff `--fix` or manually where noqa was not appropriate. No functional impact.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Package skeleton is complete and stable; Plans 02 and 03 can proceed
- `src/bitbat/cli/commands/` directory exists and is ready to receive command modules
- All import paths verified; no circular imports; lint-imports contract satisfied

## Self-Check: PASSED

- FOUND: src/bitbat/cli/__init__.py
- FOUND: src/bitbat/cli/_helpers.py
- FOUND: src/bitbat/cli/commands/__init__.py
- CONFIRMED: src/bitbat/cli.py deleted
- FOUND: e3df7ed (feat: extract helpers)
- FOUND: 163c0db (feat: package conversion)
- FOUND: 24690ba (chore: verification)

---
*Phase: 32-cli-decomposition*
*Completed: 2026-03-12*
