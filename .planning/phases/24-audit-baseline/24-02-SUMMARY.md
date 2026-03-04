---
phase: 24-audit-baseline
plan: 02
subsystem: testing
tags: [vulture, radon, dead-code, cyclomatic-complexity, static-analysis, audit]

# Dependency graph
requires:
  - phase: 24-01
    provides: "Test classification and coverage gap matrix (establishes audit baseline structure)"
provides:
  - "Triaged vulture dead code report with false-positive whitelist"
  - "Radon complexity report with all rank C+ functions mapped to v1.5 requirements"
  - "Evidence that src/bitbat has zero dead code at 80% confidence"
  - "31 high-complexity functions identified as remediation candidates"
affects: [24-03, 25-correctness, 26-architecture, 27-ci-gates]

# Tech tracking
tech-stack:
  added: []
  patterns: ["vulture whitelist workflow for repeatable dead-code scans", "radon rank-filtered complexity reports with requirement annotations"]

key-files:
  created:
    - ".planning/phases/24-audit-baseline/evidence/vulture.txt"
    - ".planning/phases/24-audit-baseline/evidence/vulture-whitelist.py"
    - ".planning/phases/24-audit-baseline/evidence/vulture-triaged.txt"
    - ".planning/phases/24-audit-baseline/evidence/radon.txt"
  modified: []

key-decisions:
  - "All 17 vulture findings are pytest fixture false positives; zero genuine dead code in src/bitbat"
  - "5 cli.py functions (CC 15-37) mapped to DEBT-01 (deferred v1.6+), not v1.5 remediation scope"
  - "LivePredictor.predict_latest (CC=28) identified as highest non-CLI complexity, maps to ARCH-06"

patterns-established:
  - "Vulture whitelist: .planning/phases/24-audit-baseline/evidence/vulture-whitelist.py as repeatable false-positive exclusion file"
  - "Radon summary header: prepend rank counts, top-5, and requirement mapping to raw tool output"

requirements-completed: [AUDT-02, AUDT-04]

# Metrics
duration: 3min
completed: 2026-03-04
---

# Phase 24 Plan 02: Dead Code & Complexity Audit Summary

**Vulture dead code scan (zero genuine findings after whitelist triage) and radon complexity audit identifying 31 functions at CC >= 11 across 15 modules**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-04T13:24:13Z
- **Completed:** 2026-03-04T13:27:26Z
- **Tasks:** 2
- **Files created:** 4

## Accomplishments
- Ran vulture at 80% confidence: 17 raw findings, all confirmed as pytest fixture false positives, zero genuine dead code in src/bitbat
- Created curated whitelist (5 entries) with comments explaining each false positive category for repeatable future scans
- Ran radon CC scan: 31 functions at rank C+ (1 rank E, 6 rank D, 24 rank C) with full requirement mapping
- Confirmed research predictions: model_cv (CC=37), predict_latest (CC=28), model_optimize (CC=27) are the top three

## Task Commits

Each task was committed atomically:

1. **Task 1: Vulture dead code scan with whitelist triage** - `29cccfc` (chore)
2. **Task 2: Radon cyclomatic complexity audit** - `3165f02` (chore)

## Files Created
- `.planning/phases/24-audit-baseline/evidence/vulture.txt` - Raw vulture output (17 findings, all in test files)
- `.planning/phases/24-audit-baseline/evidence/vulture-whitelist.py` - Curated whitelist of 5 confirmed false positives with category comments
- `.planning/phases/24-audit-baseline/evidence/vulture-triaged.txt` - Annotated triage report showing 0 genuine dead code after whitelist
- `.planning/phases/24-audit-baseline/evidence/radon.txt` - Radon CC report with summary header: rank counts, top-5, requirement mapping, and raw output

## Decisions Made
- **All 17 vulture findings whitelisted:** Every finding was a pytest fixture parameter or lambda stub parameter. No borderline cases -- all are well-understood false positive patterns (dependency injection, monkeypatch stubs).
- **CLI monolith complexity deferred:** The 5 cli.py functions (model_cv CC=37, model_optimize CC=27, monitor_run_once CC=19, features_build CC=18, batch_run CC=15) map to DEBT-01 which is explicitly deferred to v1.6+. They will be flagged by the ARCH-06 ruff gate but are not remediation targets in v1.5.
- **predict_latest is the actionable target:** At CC=28, LivePredictor.predict_latest is the highest-complexity function outside the CLI monolith and directly relevant to the autonomous prediction flow (core value).

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Key Findings for Downstream Phases

### Dead Code (AUDT-02)
- **Result:** Clean. No dead code removal needed.
- **Implication:** Phase 25-27 remediation work does not need to account for dead code cleanup.

### Complexity (AUDT-04)
- **Result:** 31 functions exceed CC 10 (the ARCH-06 threshold).
- **Top priority targets for v1.5:**
  - `LivePredictor.predict_latest` (CC=28, rank D) -- autonomous predictor, core value
  - `ensure_feature_contract` (CC=22, rank D) -- contracts, touched by CORR-03/CORR-04
  - `prediction_performance` (CC=23, rank D) -- API route
  - `select_champion_report` (CC=23, rank D) -- model evaluation
  - `fetch` in news_cryptocompare (CC=23, rank D) -- ingestion
- **Deferred targets (DEBT-01):** 5 cli.py functions totaling CC 15-37

## Next Phase Readiness
- Evidence files ready for 24-03 (coverage report and E2E smoke test)
- Dead code and complexity baselines established for AUDIT-REPORT.md synthesis
- Whitelist file available for future vulture re-runs after code changes

## Self-Check: PASSED

- All 4 evidence files exist
- All 1 summary file exists
- Commit 29cccfc (Task 1) verified in git log
- Commit 3165f02 (Task 2) verified in git log

---
*Phase: 24-audit-baseline*
*Completed: 2026-03-04*
