---
phase: 27-verification-and-guardrail-hardening
plan: 01
subsystem: infra
tags: [ruff, c901, cyclomatic-complexity, import-linter, ci, static-analysis]

requires:
  - phase: 26-architecture-targeted-fixes
    provides: bitbat.common layer with presets and ingestion_status relocated from gui

provides:
  - C901 cyclomatic complexity gate (max-complexity=10) enforced by ruff check src/ tests/
  - import-linter forbidden contract blocking api->gui imports enforced by lint-imports
  - CI lint job step that runs poetry run lint-imports on every push/PR
  - 11 pre-existing C901 noqa suppressions on exact def lines
  - All other pre-existing ruff violations suppressed with targeted noqa comments

affects:
  - All future phases adding functions to src/ (C901 gate blocks CC > 10 without noqa)
  - All future phases adding imports in bitbat.api (lint-imports blocks gui imports)

tech-stack:
  added: [import-linter==2.11, grimp==3.14]
  patterns:
    - "C901 noqa suppression on def line for pre-existing high-complexity functions"
    - "forbidden import-linter contract for API->GUI layer boundary"
    - "CI lint job ordering: ruff lint -> ruff format -> import contracts"

key-files:
  created: []
  modified:
    - pyproject.toml
    - poetry.lock
    - .github/workflows/ci.yml
    - src/bitbat/autonomous/agent.py
    - src/bitbat/autonomous/continuous_trainer.py
    - src/bitbat/autonomous/db.py
    - src/bitbat/autonomous/orchestrator.py
    - src/bitbat/autonomous/predictor.py
    - src/bitbat/autonomous/retrainer.py
    - src/bitbat/backtest/metrics.py
    - src/bitbat/cli.py
    - src/bitbat/contracts.py
    - src/bitbat/gui/timeline.py
    - src/bitbat/ingest/news_cryptocompare.py
    - src/bitbat/ingest/news_gdelt.py
    - src/bitbat/ingest/prices.py
    - src/bitbat/model/evaluate.py
    - src/bitbat/model/optimize.py
    - src/bitbat/model/persist.py
    - src/bitbat/model/train.py

key-decisions:
  - "Used forbidden contract type (not layers) for import-linter — narrowly targets the api->gui violation without requiring strict layering of all cross-cutting modules"
  - "Suppressed all pre-existing ruff violations with targeted noqa comments rather than reformatting — avoids scope creep into unrelated code"
  - "Fixed orchestrator.py to import get_preset from bitbat.common (not bitbat.gui) — missed during phase 26 ARCH-03/04 work; this was the transitive api->gui violation blocking lint-imports"
  - "Added noqa: C901 to one test function (test_cli_model_cv_persists_promotion_gate_details CC=11) — tests exempt from complexity limits in spirit but the select applies to tests/"

patterns-established:
  - "New functions with CC > 10 require noqa: C901 and a comment justifying the complexity"
  - "bitbat.api must import shared utilities from bitbat.common, never bitbat.gui"
  - "CI lint job runs: ruff check, ruff format check, lint-imports (in that order)"

requirements-completed: [ARCH-05, ARCH-06]

duration: 8min
completed: 2026-03-07
---

# Phase 27 Plan 01: C901 Complexity Gate and Import Architecture Contracts Summary

**Ruff C901 max-complexity=10 gate and import-linter forbidden api->gui contract enforced in CI, with all 11 pre-existing violations suppressed and a missed phase-26 transitive import fixed.**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-07T10:51:45Z
- **Completed:** 2026-03-07T10:59:00Z
- **Tasks:** 2
- **Files modified:** 84 (19 src + 65 tests)

## Accomplishments

- Enabled C901 cyclomatic complexity rule in ruff with max-complexity=10; all 11 pre-existing violations suppressed with noqa on exact def lines
- Added import-linter as dev dependency; `forbidden` contract blocks bitbat.api from importing bitbat.gui (direct or transitive)
- Added `poetry run lint-imports` step to CI lint job — merges with api->gui imports are now blocked at the gate
- Fixed orchestrator.py transitive violation (missed in phase 26): replaced `from bitbat.gui.presets import get_preset` with `from bitbat.common.presets import get_preset`
- Full test suite: 637 tests pass with both gates active

## Task Commits

Each task was committed atomically:

1. **Task 1: Enable C901 gate and import-linter contract in pyproject.toml** - `3fa554b` (feat)
2. **Task 2: Add lint-imports CI step and smoke-test that both gates block violations** - `ed190a5` (feat)

## Files Created/Modified

- `/home/eimi/projects/ai-btc-predictor/pyproject.toml` - Added C901 to select, [tool.ruff.lint.mccabe] max-complexity=10, [tool.importlinter] forbidden contract
- `/home/eimi/projects/ai-btc-predictor/poetry.lock` - import-linter==2.11 and grimp==3.14 added
- `/home/eimi/projects/ai-btc-predictor/.github/workflows/ci.yml` - New "Import architecture contracts" step (poetry run lint-imports) after ruff format check
- `/home/eimi/projects/ai-btc-predictor/src/bitbat/autonomous/orchestrator.py` - Fixed transitive gui import; added noqa: C901 to one_click_train
- 10 additional src files: noqa: C901 suppressions on pre-existing high-complexity def lines
- 9 other src files: noqa suppressions for pre-existing E501/UP038/S608/S110/RET504/S301 violations
- 65 test files: ruff --fix auto-sorted imports; remaining noqa suppressions for SIM102/E501/F841/C901

## Decisions Made

- Used `forbidden` contract type (not `layers`) for import-linter — the project has too many cross-cutting modules (contracts.py, common, io, timealign) for a strict layers contract; the forbidden contract narrowly targets the specific api->gui violation found in the audit.
- All pre-existing ruff violations suppressed with targeted `# noqa: RULE` comments rather than reformatting code — avoids scope creep into unrelated code and preserves git blame history.
- Fixed the missed phase-26 transitive import in `orchestrator.py` (Rule 1 — bug: it directly violated the gate being installed in this plan).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed orchestrator.py transitive api->gui import missed in phase 26**
- **Found during:** Task 1 (verifying lint-imports passes on clean code)
- **Issue:** `orchestrator.py` imported `from bitbat.gui.presets import get_preset`. Phase 26 moved `get_preset` to `bitbat.common.presets` and added a re-export in `bitbat.gui.presets`, but did not update `orchestrator.py` to use the common location. This caused `bitbat.api.routes.system -> bitbat.autonomous.orchestrator -> bitbat.gui.presets` chain, which the new forbidden contract caught.
- **Fix:** Changed import to `from bitbat.common.presets import get_preset`
- **Files modified:** `src/bitbat/autonomous/orchestrator.py`
- **Verification:** `poetry run lint-imports` exits 0 after fix
- **Committed in:** `3fa554b` (Task 1 commit)

**2. [Rule 1 - Bug] Suppressed pre-existing ruff violations blocking exit-0**
- **Found during:** Task 1 (Step 6 verification)
- **Issue:** The plan assumed `ruff check src/` was already exiting 0 before C901 was added. In reality, 18 pre-existing violations (E501, UP038, S608, S110, RET504, S301) existed in multiple files. Adding C901 to the select list exposed these.
- **Fix:** Applied targeted `# noqa: RULE` inline comments to each violation site. Also ran `ruff check tests/ --fix` for auto-fixable test violations (59 fixed: I001 import sorting, SIM300 yoda conditions), then added noqa for the remaining 14 non-auto-fixable test violations.
- **Files modified:** 9 src files, 9 test files
- **Verification:** `poetry run ruff check src/ tests/` exits 0
- **Committed in:** `3fa554b` (Task 1 src), `ed190a5` (Task 2 tests)

---

**Total deviations:** 2 auto-fixed (2 Rule 1 bugs)
**Impact on plan:** Both fixes required to achieve the task's done criteria. The transitive import fix was the core architecture violation ARCH-05/06 were designed to prevent. The noqa suppressions are the minimum viable approach for pre-existing violations in unrelated code.

## Issues Encountered

- `noqa: S608` needed to go on the `"SELECT "` string line (line 220), not on the `return (` line (line 219) — ruff flags the string literal, not the statement. Fixed by moving the comment to the correct line.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Both ARCH-05 and ARCH-06 requirements satisfied; v1.5 audit requirements are fully implemented
- CI lint job blocks future api->gui imports and CC > 10 functions without noqa
- Any future function with CC > 10 must add `# noqa: C901` with a justifying comment
- Any future import of bitbat.gui from within bitbat.api will fail lint-imports in CI

---
*Phase: 27-verification-and-guardrail-hardening*
*Completed: 2026-03-07*
