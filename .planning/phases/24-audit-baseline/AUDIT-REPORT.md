# Phase 24: Audit Baseline Report

**Date:** 2026-03-04
**Milestone:** v1.5 Codebase Health Audit & Critical Remediation
**Scope:** Full codebase health assessment of bitbat (6,406 statements, 76 test files, 603 tests)

This report synthesizes all evidence gathered during phase 24 (plans 01-03) into a single document that drives planning for phases 25-27.

---

## Section 1: Severity-Sorted Summary Table

| # | Finding | Severity | Requirement | Source |
|---|---------|----------|-------------|--------|
| 1 | Retrainer passes `--tau` to `features build` which does not accept it; causes `UsageError` at autonomous retraining time | CRITICAL | CORR-01 | research:code-inspection |
| 2 | CV metric key naming is fragile: `_read_cv_score()` uses a 3-level cascade (`average_balanced_accuracy` -> `champion.mean_directional_accuracy` -> `1 - average_rmse`); key naming mismatch between concepts could silently return wrong score | CRITICAL | CORR-02 | research:code-inspection |
| 3 | `autonomous/continuous_trainer.py` has 17% branch coverage -- nearly untested production code that runs the retraining pipeline | HIGH | AUDT-03 | tool:pytest-cov |
| 4 | `autonomous/alerting.py` has 25% branch coverage -- alert delivery logic is nearly untested | HIGH | AUDT-03 | tool:pytest-cov |
| 5 | `autonomous/predictor.py` has 26% branch coverage -- live prediction logic is largely untested despite being core value | HIGH | AUDT-03 | tool:pytest-cov |
| 6 | API `routes/system.py` imports from `gui/widgets` and `gui/presets` (3 import sites) -- cross-layer dependency | HIGH | ARCH-04 | research:code-inspection |
| 7 | `api/routes/system.py` has 46% branch coverage -- settings/system management is half-untested | HIGH | AUDT-03 | tool:pytest-cov |
| 8 | All 14 v1.5 requirements confirmed as test coverage gaps -- no existing test exercises the code paths these requirements target | HIGH | AUDT-01 | tool:test-classification |
| 9 | `LivePredictor.predict_latest` has CC=28 (rank D) -- highest-complexity function outside CLI monolith, core autonomous prediction path | HIGH | ARCH-06 | tool:radon |
| 10 | `model train` output path does not respect `data_dir` from config -- model saved to project root `models/` regardless of config | HIGH | DEFER | smoke-test |
| 11 | `autonomous.db` path partially hardcoded -- `batch run` stores predictions in both `data_dir/predictions/` (correct) and `data/autonomous.db` (project root) | HIGH | DEFER | smoke-test |
| 12 | `macro_ingestion.py` and `onchain_ingestion.py` both have 0% coverage -- completely untested modules | MEDIUM | AUDT-03 | tool:pytest-cov |
| 13 | `autonomous/news_ingestion.py` has 56% branch coverage | MEDIUM | AUDT-03 | tool:pytest-cov |
| 14 | `autonomous/retrainer.py` has 57% branch coverage -- partially tested but critical retraining logic has gaps | MEDIUM | AUDT-03 | tool:pytest-cov |
| 15 | `regression_metrics()` mixes computation with file I/O side effects | MEDIUM | CORR-03 | research:code-inspection |
| 16 | `assert isinstance` used in production code paths (survives only without `-O` flag) | MEDIUM | CORR-04 | research:code-inspection |
| 17 | `test_leakage.py` referenced in CLAUDE.md but does not exist | MEDIUM | CORR-05 | tool:test-classification |
| 18 | API route freq/horizon defaults hardcoded instead of sourced from config | MEDIUM | CORR-06 | research:code-inspection |
| 19 | Private feature pipeline functions used by 3 external callers via private API | MEDIUM | ARCH-01 | research:code-inspection |
| 20 | 3 divergent price-loading implementations | MEDIUM | ARCH-02 | research:code-inspection |
| 21 | No config reset function for test isolation | MEDIUM | ARCH-03 | research:code-inspection |
| 22 | No import-linter CI contracts | MEDIUM | ARCH-05 | research:code-inspection |
| 23 | No ruff C901 complexity gate in CI; 31 functions exceed CC 10 | MEDIUM | ARCH-06 | tool:radon |
| 24 | OBV fold-boundary leakage not assessed; cumsum computation may leak future data at fold boundaries | MEDIUM | LEAK-01, LEAK-02 | research:code-inspection |
| 25 | CLI monolith: `model_cv` CC=37 (rank E), 5 functions in cli.py total CC 15-37 | MEDIUM | DEBT-01 | tool:radon |
| 26 | Zero genuine dead code detected in src/bitbat | INFO | AUDT-02 | tool:vulture |

---

## Section 2: Test Suite Health (AUDT-01)

**Evidence:** `evidence/test-classification.md` (Plan 01)

### Summary

| Metric | Count |
|--------|-------|
| Total test files (after cleanup) | 76 |
| Total tests | 603 |
| Files deleted (pure source-reader) | 1 |
| Files reclassified (*_complete.py kept) | 16 |

### Classification by Marker

| Marker | Files | Tests | Percentage |
|--------|-------|-------|------------|
| behavioral | 32 | 269 | 44.6% |
| integration | 37 | 308 | 51.1% |
| structural | 7 | 26 | 4.3% |
| **Total** | **76** | **603** | **100%** |

### Key Actions Taken (Plan 01)

- Deleted `test_phase19_d1_monitor_alignment_complete.py` (5 pure source-reader tests)
- Reclassified 16 `*_complete.py` files that exercise real production code (kept as integration/structural)
- Registered behavioral/integration/structural markers in `pyproject.toml`
- Cleaned up references to deleted file from 2 dependent test files and Makefile

### Coverage Gap Matrix

All 14 v1.5 requirements are confirmed coverage gaps:

| Category | Requirements | Covered | Gap |
|----------|-------------|---------|-----|
| CORR (Correctness) | 6 (CORR-01 through CORR-06) | 0 | 6 |
| LEAK (Leakage) | 2 (LEAK-01, LEAK-02) | 0 | 2 |
| ARCH (Architecture) | 6 (ARCH-01 through ARCH-06) | 0 | 6 |
| **Total** | **14** | **0** | **14** |

This is expected: the requirements were defined to target known issues, and the test suite does not cover these specific code paths.

---

## Section 3: Dead Code Findings (AUDT-02)

**Evidence:** `evidence/vulture.txt`, `evidence/vulture-whitelist.py`, `evidence/vulture-triaged.txt` (Plan 02)

### Summary

| Metric | Count |
|--------|-------|
| Raw vulture findings (80% confidence) | 17 |
| Whitelisted (confirmed false positives) | 17 |
| Genuine dead code | 0 |

### False Positive Breakdown

| Category | Count | Source Files |
|----------|-------|-------------|
| Pytest fixture parameters | 15 | tests/api/test_metrics.py, tests/api/test_predictions.py |
| Lambda stub parameters | 2 | tests/autonomous/test_orchestrator.py |

### Whitelist

5 entries in `evidence/vulture-whitelist.py`:
- `db_with_data` -- pytest fixture
- `incompatible_schema_db` -- pytest fixture
- `db_with_predictions` -- pytest fixture
- `model_on_disk` -- pytest fixture
- `kw` -- lambda stub parameter

### Requirement Mapping

No findings to map. The codebase has zero dead code at the 80% confidence threshold.

**Note:** Vulture detects structurally unreachable code via AST analysis. It does NOT detect semantic bugs like wrong CLI arguments (CORR-01) or key naming fragility (CORR-02). Those require manual code review or integration testing.

---

## Section 4: Complexity Findings (AUDT-04)

**Evidence:** `evidence/radon.txt` (Plan 02)

### Summary

| Rank | CC Range | Count | Risk Level |
|------|----------|-------|------------|
| F | 41+ | 0 | Error-prone |
| E | 31-40 | 1 | Very high -- priority fix |
| D | 21-30 | 6 | High -- remediation candidate |
| C | 11-20 | 24 | Moderate -- flag for attention |
| **Total C+** | **11+** | **31** | |

### Top 10 Highest-Complexity Functions

| # | Function | File | CC | Rank | Requirement |
|---|----------|------|----|------|-------------|
| 1 | `model_cv` | cli.py:544 | 37 | E | DEBT-01 (DEFER) |
| 2 | `LivePredictor.predict_latest` | autonomous/predictor.py:142 | 28 | D | ARCH-06 |
| 3 | `model_optimize` | cli.py:852 | 27 | D | DEBT-01 (DEFER) |
| 4 | `fetch` (news_cryptocompare) | ingest/news_cryptocompare.py:259 | 23 | D | DEFER |
| 5 | `prediction_performance` | api/routes/predictions.py:78 | 23 | D | DEFER |
| 6 | `select_champion_report` | model/evaluate.py:381 | 23 | D | DEFER |
| 7 | `ensure_feature_contract` | contracts.py:79 | 22 | D | CORR-03/04 |
| 8 | `one_click_train` | autonomous/orchestrator.py:29 | 19 | C | DEFER |
| 9 | `monitor_run_once` | cli.py:1466 | 19 | C | DEBT-01 (DEFER) |
| 10 | `features_build` | cli.py:396 | 18 | C | DEBT-01 (DEFER) |

### Requirement Mapping

**v1.5 remediation targets (non-deferred):**
- `LivePredictor.predict_latest` (CC=28) -- highest non-CLI complexity, core autonomous path
- `ensure_feature_contract` (CC=22) -- touched by CORR-03/CORR-04 fixes
- `AutoRetrainer.retrain` (CC=12) -- CORR-01 fix site
- `MonitoringAgent.run_once` (CC=15) -- may simplify during CORR-01/02 fixes
- `build_xy` (CC=15) -- LEAK-02 fix site
- `fit_baseline` (CC=12) -- may be touched by CORR-05
- `WalkForwardResult.summary` (CC=18) -- walk-forward CV

**Deferred (DEBT-01, v1.6+):**
- 5 cli.py functions (CC 15-37): `model_cv`, `model_optimize`, `monitor_run_once`, `features_build`, `batch_run`

---

## Section 5: Branch Coverage (AUDT-03)

**Evidence:** `evidence/coverage.txt` (Plan 03, this plan)

### Overall

| Metric | Value |
|--------|-------|
| Total statements | 6,406 |
| Missed statements | 1,265 |
| Total branches | 1,518 |
| Partial branches | 348 |
| **Overall branch coverage** | **77%** |
| Tests passed | 603 |

### Bottom 15 Lowest-Coverage Modules

| # | Module | Stmts | Miss | Branch | BrPart | Cover |
|---|--------|-------|------|--------|--------|-------|
| 1 | `autonomous/macro_ingestion.py` | 48 | 48 | 10 | 0 | **0%** |
| 2 | `autonomous/onchain_ingestion.py` | 48 | 48 | 10 | 0 | **0%** |
| 3 | `autonomous/continuous_trainer.py` | 150 | 119 | 30 | 0 | **17%** |
| 4 | `autonomous/alerting.py` | 94 | 68 | 22 | 3 | **25%** |
| 5 | `autonomous/predictor.py` | 164 | 114 | 40 | 2 | **26%** |
| 6 | `api/routes/system.py` | 152 | 80 | 36 | 4 | **46%** |
| 7 | `autonomous/news_ingestion.py` | 130 | 52 | 26 | 4 | **56%** |
| 8 | `autonomous/retrainer.py` | 149 | 57 | 28 | 1 | **57%** |
| 9 | `analytics/feature_analysis.py` | 73 | 21 | 22 | 2 | **65%** |
| 10 | `gui/widgets.py` | 204 | 57 | 82 | 14 | **69%** |
| 11 | `autonomous/orchestrator.py` | 119 | 31 | 20 | 6 | **69%** |
| 12 | `autonomous/validator.py` | 190 | 50 | 62 | 22 | **70%** |
| 13 | `ingest/news_cryptocompare.py` | 190 | 48 | 66 | 22 | **70%** |
| 14 | `autonomous/agent.py` | 166 | 42 | 30 | 8 | **74%** |
| 15 | `ingest/news_gdelt.py` | 166 | 33 | 50 | 16 | **75%** |

### Coverage Pattern Analysis

The lowest-coverage modules cluster heavily in the `autonomous/` package:
- 8 of the bottom 15 modules are in `autonomous/`
- This package contains the core value path (prediction, retraining, monitoring)
- Two modules (`macro_ingestion.py`, `onchain_ingestion.py`) have 0% coverage

### Cross-Reference with Test Classification

Low-coverage modules vs behavioral test existence:

| Module | Coverage | Has Behavioral Tests? | Has Integration Tests? |
|--------|----------|-----------------------|----------------------|
| autonomous/continuous_trainer.py | 17% | No | No |
| autonomous/alerting.py | 25% | No | No |
| autonomous/predictor.py | 26% | No | No |
| api/routes/system.py | 46% | No | Yes (partial, via test_settings.py) |
| autonomous/news_ingestion.py | 56% | No | Yes (test_session4_complete.py) |
| autonomous/retrainer.py | 57% | Yes (test_retrainer.py) | No |
| analytics/feature_analysis.py | 65% | Yes (test_feature_analysis.py) | No |
| gui/widgets.py | 69% | No | Yes (test_widgets.py, test_complete_gui.py) |

The lowest-coverage modules are also the ones missing behavioral tests. The test suite exercises these paths only indirectly through integration tests (if at all).

---

## Section 6: E2E Pipeline Smoke Test (AUDT-05)

**Evidence:** `evidence/smoke-test.log`, `evidence/smoke-summary.md` (Plan 03, this plan)

### Configuration

- Config: `freq=1h`, `horizon=4h`, `tau=0.005`
- Auxiliary features disabled (sentiment, garch, macro, onchain)
- Data source: Real yfinance data (BTC-USD, start 2025-01-01)
- Isolation: `/tmp/smoke_test/data` via `--config` data_dir

### Results

| Stage | Command | Result | Detail |
|-------|---------|--------|--------|
| 1. Ingest prices | `prices pull --symbol BTC-USD` | PASS | 10,106 rows ingested |
| 2. Build features | `features build` | PASS | 10,059-row feature matrix |
| 3. Train model | `model train` | PASS | XGBoost model saved |
| 4. Batch prediction | `batch run` | PASS | Prediction stored |
| 5. Monitor run-once | `monitor run-once` | PARTIAL | Monitor cycle completed; retraining failed (insufficient data: 10,059 < 17,280 required) |

**Overall: 4/5 stages fully pass, 1 partial pass.**

### Smoke Test Findings

1. **Model output path partially hardcoded:** `model train` saves to `models/1h_4h/xgb.json` relative to project root, not to the `data_dir` specified in config. This means the model persistence path is not fully configurable. Maps to DEFER (path centralization, DEBT-02).

2. **Autonomous DB path partially hardcoded:** `batch run` correctly stores predictions under `data_dir/predictions/` but also writes to `data/autonomous.db` at the project root. Maps to DEFER (DEBT-02/DEBT-03).

3. **Monitor retraining data requirement:** The walk-forward CV windows require approximately 17,280 bars (~2 years of 1h data). The smoke test had only ~14 months. The monitor itself completed its full monitoring cycle (validations, drift detection, prediction), but the retraining sub-step failed. This is expected behavior for a short dataset.

### Known Gap: CORR-01

The `model train` command succeeds when called directly. The CORR-01 bug only manifests when the autonomous retrainer calls `features build` via subprocess with the `--tau` argument (which `features build` does not accept). This is NOT testable via direct CLI invocation.

**Precise location:** `src/bitbat/autonomous/retrainer.py` lines 194-202:
```python
self._run_command([
    "poetry", "run", "bitbat", "features", "build",
    "--tau", str(self.tau),   # <-- features build has no --tau option
])
```

### Known Gap: CORR-02

The CV key mismatch was not exercisable in the smoke test because retraining failed before reaching the CV score comparison. From code inspection:

- The CLI (`cli.py:781`) writes `average_balanced_accuracy` to `cv_summary.json` with the value of `mean_directional_accuracy`
- The retrainer (`retrainer.py:71`) reads `average_balanced_accuracy` first, so the primary key lookup does match
- However, the naming is confusing (the key says "balanced accuracy" but stores directional accuracy) and the 3-level cascade fallback (`average_balanced_accuracy` -> `champion.mean_directional_accuracy` -> `1 - average_rmse`) is fragile

**Updated severity assessment:** CORR-02 is less severe than originally estimated. The primary key lookup works correctly. The main risk is the confusing naming and fragile cascade, which could silently break if the CLI output format changes. Recommend keeping as HIGH rather than CRITICAL.

---

## Section 7: Tool vs Research Cross-Reference

This section compares what automated tools found versus what manual code inspection (research phase) found.

| Issue | Found by Tool? | Found by Research? | Notes |
|-------|---------------|--------------------|-------|
| CORR-01: Retrainer `--tau` bug | No (vulture) | Yes | Semantic bug: wrong CLI arguments are not structurally detectable. Vulture finds unused code, not wrong usage. |
| CORR-02: CV key naming fragility | No (any tool) | Yes | Requires understanding of data flow between writer (cli.py) and reader (retrainer.py). No static analysis tool traces cross-module key consistency. |
| CORR-03: `regression_metrics()` side effects | No | Yes | Design issue; no tool detects function purity violations. |
| CORR-04: `assert isinstance` in production | Partially (ruff could) | Yes | ruff has rules for production assert usage, but it's not configured. |
| CORR-05: Missing `test_leakage.py` | Yes (test-classification gap matrix) | Yes | The gap matrix identified this as a coverage gap. |
| CORR-06: API hardcoded defaults | No | Yes | Requires cross-referencing API routes against config file. |
| ARCH-04: API->GUI import | Partially (import-linter could) | Yes | import-linter would catch this, but it's not configured. Tools like vulture or radon don't analyze import boundaries. |
| LEAK-01/02: OBV fold leakage | No | Yes | Requires domain knowledge about cumulative sum behavior at fold boundaries. |
| Low coverage modules | Yes (pytest-cov) | Yes | Coverage report identified exact same modules research predicted. |
| Dead code | Yes (vulture: none found) | Yes (predicted none) | Research correctly predicted zero genuine dead code. |
| High complexity | Yes (radon) | Yes | Research correctly identified the top 3 functions. |

### Meta-Finding: What Automated Tools Miss

Automated tools found:
- **Coverage gaps** (pytest-cov) -- accurate, quantitative
- **Dead code** (vulture) -- accurate, zero false negatives
- **Complexity hotspots** (radon) -- accurate, comprehensive

Automated tools missed:
- **Semantic bugs** (CORR-01, CORR-02) -- wrong arguments, key mismatches
- **Design issues** (CORR-03, CORR-04) -- purity violations, unsafe assert usage
- **Cross-module data flow** (CORR-06) -- config vs hardcoded value divergence
- **Domain-specific leakage** (LEAK-01/02) -- requires understanding of ML fold boundaries
- **Import boundary violations** (ARCH-04) -- requires policy definition, not just analysis

**Conclusion:** Static analysis tools are effective for quantitative health metrics (coverage, complexity, dead code) but miss semantic correctness bugs and design issues. Manual code review remains essential for finding the CRITICAL-severity issues in this codebase.

---

## Section 8: Recommendations for Phases 25-27

### Priority Order (by severity and impact)

#### Phase 25: Critical Correctness Remediation

**Must-fix first (CRITICAL/HIGH):**

1. **CORR-01 (CRITICAL):** Fix `--tau` argument in `retrainer.py`. Remove the `--tau` argument from the `features build` subprocess command. Add a test that verifies the subprocess argument list matches the actual CLI interface.

2. **CORR-02 (HIGH, downgraded from CRITICAL):** Fix CV metric key naming. Rename `average_balanced_accuracy` to `mean_directional_accuracy` consistently across writer (cli.py) and reader (retrainer.py). Simplify the 3-level cascade. Add a round-trip consistency test.

3. **CORR-05 (MEDIUM):** Create `test_leakage.py` with PR-AUC guardrail. This test is referenced in CLAUDE.md but does not exist.

4. **CORR-03 (MEDIUM):** Refactor `regression_metrics()` to separate computation from I/O.

5. **CORR-04 (MEDIUM):** Replace `assert isinstance` with proper `if not isinstance: raise TypeError` in production paths.

6. **CORR-06 (MEDIUM):** Source API freq/horizon defaults from `default.yaml`.

7. **LEAK-01/LEAK-02 (MEDIUM):** Assess OBV fold-boundary leakage empirically, fix if confirmed material.

#### Phase 26: Architecture Targeted Fixes

8. **ARCH-04 (HIGH):** Eliminate `api/routes/system.py` imports from `gui/`. Move shared utilities to a lower layer.

9. **ARCH-01 (MEDIUM):** Promote private feature pipeline functions to public API.

10. **ARCH-02 (MEDIUM):** Consolidate 3 price-loading implementations.

11. **ARCH-03 (MEDIUM):** Add config reset function for test isolation.

#### Phase 27: Verification & Guardrail Hardening

12. **ARCH-05 (MEDIUM):** Add import-linter CI contracts.

13. **ARCH-06 (MEDIUM):** Add ruff C901 complexity gate (max-complexity=10). Note: this will flag 31 existing functions. Existing functions should be baselined or exempted, with the gate preventing new violations.

### Scope Observations

1. **CORR-02 severity downgrade:** The primary key lookup works correctly. The risk is naming confusion and fragile cascading, not silent data corruption. Recommend downgrading from CRITICAL to HIGH for planning purposes.

2. **Coverage focus for phase 25:** The `autonomous/` package has the worst coverage (0-57% for critical modules) and contains the most CRITICAL bugs. Fixing CORR-01/02 should naturally improve coverage in this area.

3. **Smoke test uncovered DEFER items:** The path hardcoding findings (model output, autonomous.db) map to DEBT-02/DEBT-03 which are already deferred to v1.6+. No scope change needed for v1.5.

4. **Dead code is clean:** No dead code removal work needed in any phase. This simplifies planning.

5. **CLI complexity is deferred:** The 5 highest-complexity functions are all in `cli.py` and map to DEBT-01 (v1.6+). Phase 27's ruff gate should exempt these with a per-function noqa or a higher threshold for cli.py.

---

## Appendix: Evidence File Index

| File | Plan | Contents |
|------|------|----------|
| `evidence/test-classification.md` | 24-01 | Full test classification report, deletion/reclassification logs, coverage gap matrix |
| `evidence/vulture.txt` | 24-02 | Raw vulture output (17 findings, all false positives) |
| `evidence/vulture-whitelist.py` | 24-02 | Curated whitelist (5 entries) for repeatable scans |
| `evidence/vulture-triaged.txt` | 24-02 | Annotated triage report (0 genuine dead code) |
| `evidence/radon.txt` | 24-02 | Radon CC report (31 functions at rank C+) with summary header |
| `evidence/coverage.txt` | 24-03 | pytest-cov branch coverage report (77% overall, per-module detail) |
| `evidence/smoke-test.log` | 24-03 | Raw console output from 5-stage E2E pipeline smoke test |
| `evidence/smoke-summary.md` | 24-03 | Structured pass/fail table with error analysis |

---
*Phase: 24-audit-baseline*
*Generated: 2026-03-04*
*Synthesized from: Plans 01 (test classification), 02 (vulture + radon), 03 (coverage + smoke test), and research phase code inspection findings*
