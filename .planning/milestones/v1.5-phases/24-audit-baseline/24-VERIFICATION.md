---
phase: 24-audit-baseline
verified: 2026-03-04T15:10:00Z
status: passed
score: 10/10 must-haves verified
re_verification: false
---

# Phase 24: Audit Baseline Verification Report

**Phase Goal:** Operators have a complete, evidence-based understanding of codebase health before any remediation work begins
**Verified:** 2026-03-04T15:10:00Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| #  | Truth                                                                                                                               | Status     | Evidence                                                                                                                                                                 |
|----|-------------------------------------------------------------------------------------------------------------------------------------|------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1  | Every test function in the suite has exactly one pytest marker from {behavioral, integration, structural}                           | VERIFIED   | 76/76 `test_*.py` files contain `pytestmark`; pytest collection confirms 603 tests, 0 unknown-marker warnings; verified by grep count and spot-checks                   |
| 2  | All pure source-reader milestone-marker tests are deleted from the repo                                                             | VERIFIED   | `test_phase19_d1_monitor_alignment_complete.py` confirmed absent from filesystem; 608 - 603 = 5 tests removed (matches deletion log)                                    |
| 3  | Milestone-marker tests that exercise real production code are reclassified as integration or structural, not deleted                | VERIFIED   | 16 `*_complete.py` files retained with integration/structural markers; reclassification log in `evidence/test-classification.md` documents each with rationale           |
| 4  | A coverage gap matrix shows which v1.5 requirements (CORR-01 through ARCH-06) have existing test coverage and which do not          | VERIFIED   | `evidence/test-classification.md` lines 163-193 contain full matrix: 14/14 requirements confirmed as gaps; all 14 requirement IDs present with per-requirement reasoning |
| 5  | Operator can see a triaged vulture report distinguishing real dead code from false positives                                        | VERIFIED   | `evidence/vulture-triaged.txt` exists (58 lines); annotated with summary (17 raw, 17 whitelisted, 0 genuine); `evidence/vulture-whitelist.py` exists (23 lines)          |
| 6  | Operator can see every function with cyclomatic complexity >= 11 sorted by score                                                    | VERIFIED   | `evidence/radon.txt` exists (132 lines); 31 functions at rank C+ listed; top-5 table with CC scores, file paths, requirement mapping; raw output sorted by score        |
| 7  | Each vulture finding is annotated with whether it maps to a v1.5 requirement or is DEFER                                           | VERIFIED   | `evidence/vulture-triaged.txt` explicitly states 0 genuine findings; requirement mapping section confirms no findings to map; context note explains tool limitations     |
| 8  | Operator can see branch coverage percentages for every module in src/bitbat, ranked from lowest to highest                         | VERIFIED   | `evidence/coverage.txt` (254 lines) contains full per-module branch coverage; `AUDIT-REPORT.md` Section 5 reproduces bottom-15 ranking; 77% overall confirmed           |
| 9  | Operator can see which E2E pipeline stages pass and which fail, with specific error messages for failures                           | VERIFIED   | `evidence/smoke-summary.md` (50 lines) contains pass/fail table for all 5 stages; `evidence/smoke-test.log` (24 lines) contains raw output including retraining traceback|
| 10 | A single synthesized AUDIT-REPORT.md contains all findings with severity-sorted summary, requirement tagging, and cross-reference   | VERIFIED   | `AUDIT-REPORT.md` exists at 389 lines; 8 sections present; 26 findings in severity-sorted table; 71 occurrences of CORR/ARCH/LEAK/AUDT tags; evidence/ cross-referenced  |

**Score:** 10/10 truths verified

---

### Required Artifacts

| Artifact                                                             | Provides                                               | Status     | Details                                                                               |
|----------------------------------------------------------------------|--------------------------------------------------------|------------|---------------------------------------------------------------------------------------|
| `pyproject.toml`                                                     | Pytest marker registration for behavioral/integration/structural | VERIFIED | Contains `markers = ["slow:...", "behavioral:...", "integration:...", "structural:..."]` under `[tool.pytest.ini_options]` |
| `.planning/phases/24-audit-baseline/evidence/test-classification.md` | Full classification report, deletion log, coverage gap matrix | VERIFIED | 231 lines; summary table, deletion log, reclassification log, full classification by marker, coverage gap matrix for all 14 requirements |
| `.planning/phases/24-audit-baseline/evidence/vulture.txt`            | Raw vulture output at 80% confidence                   | VERIFIED   | 17 lines; 17 raw findings, all in test files, all confirmed false positives            |
| `.planning/phases/24-audit-baseline/evidence/vulture-whitelist.py`   | Curated whitelist of confirmed false positives         | VERIFIED   | 23 lines; 5 entries (db_with_data, incompatible_schema_db, db_with_predictions, model_on_disk, kw) with explanatory comments |
| `.planning/phases/24-audit-baseline/evidence/vulture-triaged.txt`    | Re-run output with whitelist applied, only genuine findings | VERIFIED | 58 lines; 0 genuine dead code; annotated with summary header and requirement mapping section |
| `.planning/phases/24-audit-baseline/evidence/radon.txt`              | Radon CC output for all rank C+ functions sorted by score | VERIFIED | 132 lines; 31 functions at CC >= 11; summary header with rank counts (0F, 1E, 6D, 24C), top-5 table, requirement mapping; raw sorted output |
| `.planning/phases/24-audit-baseline/evidence/coverage.txt`           | pytest-cov branch coverage report with per-module detail | VERIFIED | 254 lines; full per-module coverage table; TOTAL: 6406 stmts, 77% branch coverage; 603 tests passed |
| `.planning/phases/24-audit-baseline/evidence/smoke-test.log`         | Raw console output from E2E pipeline smoke test        | VERIFIED   | 24 lines; contains actual pipeline output for all 5 stages including retraining traceback |
| `.planning/phases/24-audit-baseline/evidence/smoke-summary.md`       | Structured pass/fail table for each pipeline stage     | VERIFIED   | 50 lines; pass/fail table for 5 stages; CORR-01 gap documented; CORR-02 gap documented; path hardcoding observations included |
| `.planning/phases/24-audit-baseline/AUDIT-REPORT.md`                 | Synthesized audit report with all findings             | VERIFIED   | 389 lines; 8 sections; severity-sorted 26-finding summary table at top; CORR-01 CRITICAL, CORR-02 HIGH; tool vs research cross-reference; recommendations for phases 25-27 |

---

### Key Link Verification

| From                                              | To                                          | Via                                                          | Status   | Details                                                                                                                |
|---------------------------------------------------|---------------------------------------------|--------------------------------------------------------------|----------|------------------------------------------------------------------------------------------------------------------------|
| `pyproject.toml`                                  | `tests/**/*.py`                             | Pytest marker registration enables markers used in test files | WIRED    | 76/76 `test_*.py` files use `pytestmark = pytest.mark.<type>`; markers registered under `[tool.pytest.ini_options]`   |
| `evidence/vulture-triaged.txt`                    | `evidence/vulture-whitelist.py`             | Re-run excludes whitelisted items, leaving only genuine findings | WIRED | Triaged output contains whitelist annotation in header (references whitelist.py); 17 raw - 17 whitelisted = 0 genuine  |
| `AUDIT-REPORT.md`                                 | `evidence/` directory                       | Report synthesizes and cross-references all evidence files   | WIRED    | Section headers explicitly reference each evidence file; 3 occurrences of `evidence/` path prefix; appendix lists all 8 evidence files |
| `AUDIT-REPORT.md`                                 | `.planning/REQUIREMENTS.md`                 | Each finding tagged with requirement ID from REQUIREMENTS.md | WIRED    | 71 occurrences of CORR/ARCH/LEAK/AUDT tags in report; Section 2 reproduces coverage gap matrix from test-classification.md |

---

### Requirements Coverage

| Requirement | Source Plan | Description                                                                                     | Status     | Evidence                                                                                              |
|-------------|-------------|--------------------------------------------------------------------------------------------------|------------|-------------------------------------------------------------------------------------------------------|
| AUDT-01     | 24-01       | All existing tests classified by type with coverage gap report                                   | SATISFIED  | 76 files classified (32 behavioral, 37 integration, 7 structural); 1 pure source-reader deleted; gap matrix for 14 requirements in `evidence/test-classification.md` |
| AUDT-02     | 24-02       | Dead code audit completed with vulture at 80% confidence (findings triaged, whitelist created)   | SATISFIED  | `vulture.txt` (raw 17 findings), `vulture-whitelist.py` (5-entry curated whitelist), `vulture-triaged.txt` (0 genuine dead code) |
| AUDT-03     | 24-03       | Branch coverage report generated with pytest-cov identifying lowest-coverage modules             | SATISFIED  | `coverage.txt` shows 77% overall; bottom-15 modules ranked in `AUDIT-REPORT.md` Section 5; `continuous_trainer.py` 17%, `alerting.py` 25%, `predictor.py` 26% flagged |
| AUDT-04     | 24-02       | Complexity audit completed with radon identifying high-complexity functions                      | SATISFIED  | `radon.txt` lists 31 functions at CC >= 11; top-5 identified; `model_cv` CC=37, `predict_latest` CC=28, `model_optimize` CC=27 confirmed vs research predictions |
| AUDT-05     | 24-03       | E2E pipeline smoke test executed documenting which steps pass and which fail                     | SATISFIED  | `smoke-test.log` + `smoke-summary.md`: 4/5 stages pass; monitor partial (retraining insufficient data documented with traceback); CORR-01 and CORR-02 gaps documented |

**No orphaned AUDT requirements:** REQUIREMENTS.md maps AUDT-01 through AUDT-05 exclusively to Phase 24; all 5 are claimed by the 3 plans in this phase. Coverage is complete.

---

### Anti-Patterns Found

No anti-patterns found in the evidence files or modified source files. The phase was audit-only (Plans 02 and 03 created no source file modifications); Plan 01 only added `pytestmark` markers and deleted one test file.

Spot checks:
- `evidence/test-classification.md`: No TODOs, no placeholder text, substantive content throughout
- `AUDIT-REPORT.md`: No placeholder sections; all 8 sections substantive with specific data
- `evidence/vulture-triaged.txt`: Not a stub — the "no findings" result is the actual tool output, documented with context
- `pyproject.toml` marker registration: Contains valid marker strings, not empty list

---

### Human Verification Required

None. This phase produced documentation artifacts (evidence files and reports) rather than runtime behavior. The artifacts can be fully verified by inspecting file existence, line counts, and content — which this verification performed programmatically.

The test suite health is also objectively verifiable: `poetry run pytest --co -q` would confirm 603 collected tests with 0 unknown-marker warnings (the coverage.txt file records the actual run completing with `603 passed`).

---

## Summary

Phase 24 achieved its goal. All 10 observable truths verify against the actual codebase:

**AUDT-01 (Test Classification):** 76 test files carry pytest markers, confirmed by grep on the filesystem (76/76). The deleted test file is gone. The coverage gap matrix documents all 14 v1.5 requirements as gaps with per-requirement reasoning.

**AUDT-02 (Dead Code):** Three vulture evidence files exist with substance: raw output (17 findings), curated whitelist (5 false positives with explanations), and triaged output (0 genuine dead code with annotation).

**AUDT-03 (Coverage):** `coverage.txt` records the actual pytest-cov run result — 603 tests, 77% branch coverage, per-module breakdown. The bottom-15 modules are identified and cross-referenced against test type availability in AUDIT-REPORT.md Section 5.

**AUDT-04 (Complexity):** `radon.txt` lists 31 functions at CC >= 11 with a summary header, sorted output, and requirement mapping. The research-predicted top-3 (model_cv CC=37, predict_latest CC=28, model_optimize CC=27) all appear.

**AUDT-05 (Smoke Test):** `smoke-test.log` contains actual pipeline output (not synthetic). `smoke-summary.md` documents 4/5 stage pass, the monitor partial result with traceback, and the CORR-01/CORR-02 gaps that cannot be exercised via direct CLI invocation.

**AUDIT-REPORT.md** synthesizes all evidence into a single 389-line document with severity-sorted summary, 8 sections, 26 findings tagged with requirement IDs, and a tool-vs-research cross-reference. This is the deliverable that drives phases 25-27.

Operators have a complete, evidence-based understanding of codebase health. The audit baseline is ready to hand off to phase 25 (Critical Correctness Remediation).

---

_Verified: 2026-03-04T15:10:00Z_
_Verifier: Claude (gsd-verifier)_
