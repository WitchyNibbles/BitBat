---
phase: 32-cli-decomposition
plan: 02
subsystem: cli
tags: [click, refactor, decomposition, commands]

# Dependency graph
requires:
  - phase: 32-01
    provides: cli/ package skeleton with _helpers.py — prerequisite for command extraction
provides:
  - 8 command group modules in src/bitbat/cli/commands/ (prices, news, features, backtest, batch, validate, ingest, system)
  - __init__.py reduced to model + monitor groups + add_command() wiring
affects: [32-03]

# Tech tracking
tech-stack:
  added: []
  patterns: [each CLI command group has a dedicated module in commands/; add_command() wiring in __init__.py]

key-files:
  created:
    - src/bitbat/cli/commands/prices.py
    - src/bitbat/cli/commands/news.py
    - src/bitbat/cli/commands/features.py
    - src/bitbat/cli/commands/backtest.py
    - src/bitbat/cli/commands/batch.py
    - src/bitbat/cli/commands/validate.py
    - src/bitbat/cli/commands/ingest.py
    - src/bitbat/cli/commands/system.py
  modified:
    - src/bitbat/cli/__init__.py

key-decisions:
  - "32-02: run_strategy, summarize_backtest kept as bitbat.cli re-exports from original module-level imports; monkeypatch targets updated in Plan 03"
  - "32-02: features_build monkeypatch failures (bitbat.cli.build_xy) follow same pattern as batch — deferred to Plan 03"
  - "32-02: 4 test failures are all monkeypatch targeting (tests patch bitbat.cli.* but commands hold direct refs in command modules); expected per plan"

patterns-established:
  - "Command module pattern: from bitbat.cli._helpers import only what the module's commands actually use"
  - "add_command() wiring: _cli.add_command(_module_mod.group_name) after _cli group declaration"

requirements-completed: [DEBT-01]

# Metrics
duration: 16min
completed: 2026-03-12
---

# Phase 32 Plan 02: CLI Decomposition (Simple Groups) Summary

**8 simple CLI command groups extracted from `cli/__init__.py` into `commands/{group}.py` submodules, wired back via `add_command()`, reducing `__init__.py` by 662 lines**

## Performance

- **Duration:** 16 min
- **Started:** 2026-03-12T09:07:26Z
- **Completed:** 2026-03-12T09:23:46Z
- **Tasks:** 3
- **Files modified:** 9

## Accomplishments
- Created 8 command module files in `src/bitbat/cli/commands/` with verbatim group + command definitions from `__init__.py`
- Updated `__init__.py` to use `_cli.add_command()` for all 8 extracted groups; model and monitor remain inline for Plan 03
- `bitbat --help` shows all 10 command groups; `ruff C901` passes on all 8 new modules without suppressions
- `lint-imports` passes; architecture contracts upheld

## Task Commits

Each task was committed atomically:

1. **Task 1: Create 8 simple command group modules** - `b3c8183` (feat)
2. **Task 2: Wire command groups into __init__.py and remove extracted bodies** - `9fb73ab` (feat)
3. **Task 3: Run targeted test suite** - no new commit (verification only)

## Files Created/Modified
- `src/bitbat/cli/commands/prices.py` - prices group + prices_pull command
- `src/bitbat/cli/commands/news.py` - news group + news_pull command
- `src/bitbat/cli/commands/features.py` - features group + features_build command
- `src/bitbat/cli/commands/backtest.py` - backtest group + backtest_run command; re-exports run_strategy, summarize_backtest
- `src/bitbat/cli/commands/batch.py` - batch group + batch_run, batch_realize commands; re-exports generate_price_features, aggregate_sentiment, predict_bar, load_model
- `src/bitbat/cli/commands/validate.py` - validate group + validate_run command
- `src/bitbat/cli/commands/ingest.py` - ingest group + 5 ingest commands (prices-once, news-once, macro-once, onchain-once, status)
- `src/bitbat/cli/commands/system.py` - system group + system_reset command
- `src/bitbat/cli/__init__.py` - removed 8 group bodies; added add_command() wiring; -662 lines

## Decisions Made
- Kept `run_strategy`, `summarize_backtest`, `generate_price_features`, `aggregate_sentiment`, `predict_bar`, `load_model` as `bitbat.cli` namespace re-exports via original module-level imports (same approach as Plan 01 for xgb, walk_forward etc.) so existing test monkeypatches continue to exist in `bitbat.cli` namespace — the actual fix of monkeypatch targets is deferred to Plan 03
- `model_cv` local imports of `run_strategy`/`summarize_backtest` removed since they're now top-level module imports

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Removed unused `Path` import from batch.py**
- **Found during:** Task 1 (ruff check)
- **Issue:** `Path` was imported but not used in batch.py (batch commands use `_predictions_path` helper instead)
- **Fix:** Removed `from pathlib import Path` from batch.py
- **Files modified:** src/bitbat/cli/commands/batch.py
- **Verification:** ruff F401 clean
- **Committed in:** b3c8183 (Task 1 commit)

**2. [Rule 1 - Bug] Fixed F811 redefinition errors in __init__.py re-exports**
- **Found during:** Task 2 (ruff check)
- **Issue:** Re-exporting from commands.batch/backtest duplicated existing module-level imports, causing F811 redefinitions
- **Fix:** Removed the commands.batch/backtest re-export block; kept original module-level imports as the sole re-exports for monkeypatch compatibility
- **Files modified:** src/bitbat/cli/__init__.py
- **Verification:** ruff passes with no errors
- **Committed in:** 9fb73ab (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (both Rule 1 - bugs from initial implementation)
**Impact on plan:** Both fixes were necessary for ruff compliance. No scope creep.

## Issues Encountered
- 4 CLI test failures confirmed as monkeypatch targeting issues (tests patch `bitbat.cli.build_xy`, `bitbat.cli.run_strategy`, `bitbat.cli.summarize_backtest`, `bitbat.cli.generate_price_features`, `bitbat.cli.predict_bar`). Command modules hold direct references, so patches on `bitbat.cli.*` no longer intercept. These are expected per plan and will be fixed in Plan 03.
- Failing tests: `test_cli_features_build_label_mode_default_compatibility`, `test_cli_features_build_triple_barrier_label_mode`, `test_cli_backtest_cost_slippage_reports_net_and_gross`, `test_cli_batch_run`
- `test_no_private_imports_in_callers` fails because it expects `src/bitbat/cli.py` (pre-existing failure from Plan 01 converting cli.py to cli/ package)
- `test_pipeline_stage_trace` failures are pre-existing (require operator to run `system reset` + retrain)

## Next Phase Readiness
- Plan 02 complete: 8 simple groups extracted; __init__.py contains only model, monitor, _cli, and add_command() wiring
- Plan 03 ready: extract model group + monitor group from __init__.py; fix monkeypatch targets in test_cli.py

---
*Phase: 32-cli-decomposition*
*Completed: 2026-03-12*
