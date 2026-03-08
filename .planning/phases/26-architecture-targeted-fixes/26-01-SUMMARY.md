---
phase: 26-architecture-targeted-fixes
plan: 01
subsystem: dataset, io
tags: [refactoring, public-api, price-loading, feature-pipeline]

# Dependency graph
requires:
  - phase: 25-critical-correctness-remediation
    provides: Stable OBV fold-aware feature pipeline and leakage-free dataset builder
provides:
  - Public generate_price_features and join_auxiliary_features functions in bitbat.dataset.build
  - Shared load_prices function in bitbat.io.prices consolidating 3 divergent implementations
  - Re-exports from bitbat.dataset and bitbat.io packages
  - Backward-compat aliases preserving old private names
  - Structural test guard preventing regression to private imports
affects:
  - Any future phase touching feature pipeline or price loading
  - Phase 27 (further architecture work builds on stable public API)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Public feature pipeline API: generate_price_features and join_auxiliary_features are the canonical names; private aliases retained for backward compat only
    - Shared price loader: bitbat.io.prices.load_prices (glob-based, multi-file) and load_prices_for_cli (single flat-file CLI contract)
    - Thin wrapper delegation: private methods in callers now delegate to shared function rather than duplicating logic
    - AST structural guard: test_no_private_imports_in_callers uses ast.parse to enforce no private feature imports in caller files

key-files:
  created:
    - src/bitbat/io/prices.py
    - tests/dataset/test_public_api.py
  modified:
    - src/bitbat/dataset/build.py
    - src/bitbat/dataset/__init__.py
    - src/bitbat/io/__init__.py
    - src/bitbat/cli.py
    - src/bitbat/autonomous/predictor.py
    - src/bitbat/autonomous/continuous_trainer.py
    - tests/test_cli.py

key-decisions:
  - "Backward-compat aliases (_generate_price_features = generate_price_features) kept in build.py to prevent breakage from any untracked callers while public names become canonical"
  - "Two load_prices variants: load_prices (glob-based, for autonomous pipeline) and load_prices_for_cli (single flat-file, for CLI ingest convention)"
  - "Caller private wrappers (_load_ingested_prices, _load_prices, _load_prices_indexed) retained as thin delegation wrappers to minimize blast radius — they delegate to shared module rather than containing logic"
  - "AST-based structural test guards against regression: test_no_private_imports_in_callers uses ast.parse on caller files"

patterns-established:
  - "Public-first feature API: functions in dataset/build.py named without underscore are the stable contract"
  - "Shared IO module pattern: divergent data-loading implementations consolidated into bitbat.io submodule"

requirements-completed: [ARCH-01, ARCH-02]

# Metrics
duration: 8min
completed: 2026-03-07
---

# Phase 26 Plan 01: Architecture Targeted Fixes - Public API and Shared Price Loading Summary

**Promoted private _generate_price_features/_join_auxiliary_features to public API and consolidated 3 divergent price loading implementations into a single bitbat.io.prices shared module**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-07T00:46:51Z
- **Completed:** 2026-03-07T00:54:00Z
- **Tasks:** 3
- **Files modified:** 8 (plus 1 test file fixed as deviation)

## Accomplishments
- Promoted `_generate_price_features` and `_join_auxiliary_features` to public API (no underscore) with backward-compat aliases
- Created `bitbat.io.prices` module with `load_prices` (glob-based, unifies 3 divergent implementations) and `load_prices_for_cli` (CLI flat-file contract)
- Updated all 3 external callers (cli.py, predictor.py, continuous_trainer.py) to import from public API
- Added 6 structural tests including an AST-based guard against regression to private imports
- Full test suite: 632 passed

## Task Commits

Each task was committed atomically:

1. **Task 1: Promote private feature functions to public API and consolidate price loading** - `07ff270` (feat)
2. **Task 2: Update all external callers to use public API** - `e347567` (refactor)
3. **Task 3: Add tests for public API and run full suite** - `552eea8` (test)

## Files Created/Modified
- `src/bitbat/dataset/build.py` - Renamed private functions to public, added backward-compat aliases, updated internal call sites
- `src/bitbat/dataset/__init__.py` - Added re-exports of generate_price_features, join_auxiliary_features, build_xy with __all__
- `src/bitbat/io/prices.py` - New module: load_prices (glob-based multi-file) and load_prices_for_cli (CLI flat-file)
- `src/bitbat/io/__init__.py` - Re-exports load_prices
- `src/bitbat/cli.py` - Imports public generate_price_features; _load_prices_indexed delegates to load_prices_for_cli
- `src/bitbat/autonomous/predictor.py` - Imports public generate_price_features and join_auxiliary_features; _load_ingested_prices delegates to load_prices
- `src/bitbat/autonomous/continuous_trainer.py` - Imports public generate_price_features; _load_prices delegates to load_prices
- `tests/dataset/test_public_api.py` - 6 structural tests for public API and regression guard
- `tests/test_cli.py` - Fixed monkeypatch path from _generate_price_features to generate_price_features (deviation fix)

## Decisions Made
- Backward-compat aliases retained in build.py so any untracked caller using old names continues to work during a migration window
- Two load_prices variants chosen: the autonomous pipeline needs glob-based multi-file scan; the CLI needs the explicit single flat-file contract with a ClickException on missing file
- Caller wrappers kept as thin delegation methods to minimize change surface — the private name is preserved, only the implementation changes

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed test_cli_batch_run monkeypatch path after function rename**
- **Found during:** Task 3 (run full test suite)
- **Issue:** `tests/test_cli.py::test_cli_batch_run` used `monkeypatch.setattr("bitbat.cli._generate_price_features", ...)` which raised AttributeError because the function was renamed to `generate_price_features` in Task 1
- **Fix:** Changed monkeypatch path to `bitbat.cli.generate_price_features`
- **Files modified:** tests/test_cli.py
- **Verification:** `poetry run pytest tests/test_cli.py::test_cli_batch_run` passed; full suite 632/632
- **Committed in:** 552eea8 (Task 3 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - Bug)
**Impact on plan:** Necessary fix to keep test suite consistent with the rename. No scope creep.

## Issues Encountered
- Pre-existing ruff E501 (line-too-long) errors exist in continuous_trainer.py (lines 40, 43) and cli.py (line 1498). These were present before this plan and are out-of-scope per deviation scope rules.

## Next Phase Readiness
- Public feature API is stable; future callers should import from `bitbat.dataset.build` or `bitbat.dataset` directly
- Price loading is centralized in `bitbat.io.prices`; future callers should use `load_prices` or `load_prices_for_cli`
- Structural test guard in `tests/dataset/test_public_api.py` will fail CI if any caller regresses to private imports

## Self-Check: PASSED

All files confirmed present:
- FOUND: src/bitbat/dataset/build.py
- FOUND: src/bitbat/dataset/__init__.py
- FOUND: src/bitbat/io/prices.py
- FOUND: tests/dataset/test_public_api.py
- FOUND: 26-01-SUMMARY.md

All commits confirmed:
- FOUND: 07ff270 feat(26-01): promote private feature functions to public API
- FOUND: e347567 refactor(26-01): update all external callers to use public feature API
- FOUND: 552eea8 test(26-01): add public API structural tests

---
*Phase: 26-architecture-targeted-fixes*
*Completed: 2026-03-07*
