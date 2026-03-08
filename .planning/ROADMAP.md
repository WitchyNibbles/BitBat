# Roadmap: BitBat Reliability and Timeline Evolution

## Milestones

- ✅ **v1.0 Reliability and Timeline Evolution** — Phases 1-9 shipped on 2026-02-25 ([roadmap archive](milestones/v1.0-ROADMAP.md), [requirements archive](milestones/v1.0-REQUIREMENTS.md), [audit archive](milestones/v1.0-MILESTONE-AUDIT.md)).
- ✅ **v1.1 UI-First Simplification** — Phases 10-12 shipped on 2026-02-25 ([roadmap archive](milestones/v1.1-ROADMAP.md), [requirements archive](milestones/v1.1-REQUIREMENTS.md), [audit archive](milestones/v1.1-MILESTONE-AUDIT.md)).
- ✅ **v1.2 BTC Prediction Accuracy Evolution** — Phases 13-16 shipped on 2026-02-26 ([roadmap archive](milestones/v1.2-ROADMAP.md), [requirements archive](milestones/v1.2-REQUIREMENTS.md)).
- ✅ **v1.3 Autonomous Monitor Alignment and Metrics Integrity** — Phases 17-19 shipped on 2026-02-26 ([roadmap archive](milestones/v1.3-ROADMAP.md), [requirements archive](milestones/v1.3-REQUIREMENTS.md)).
- ✅ **v1.4 Configuration Alignment** — Phases 20-23 shipped on 2026-03-01 ([roadmap archive](milestones/v1.4-ROADMAP.md), [requirements archive](milestones/v1.4-REQUIREMENTS.md), [audit archive](milestones/v1.4-MILESTONE-AUDIT.md)).

## Phases

### v1.5 Codebase Health Audit & Critical Remediation

- [x] **Phase 24: Audit Baseline** - Classify test suite, measure coverage and complexity, run dead code detection, and execute E2E pipeline smoke test to establish a trustworthy baseline before any fixes (completed 2026-03-04)
- [x] **Phase 25: Critical Correctness Remediation** - Fix silently broken production paths (retrainer CLI contract, CV metric keys, OBV leakage) and close missing guardrails (test_leakage.py, assert guards, API defaults) (completed 2026-03-06)
- [x] **Phase 26: Architecture Targeted Fixes** - Promote private APIs, consolidate duplicated logic, add config reset for test isolation, and eliminate cross-layer imports (completed 2026-03-07)
- [x] **Phase 27: Verification & Guardrail Hardening** - Add CI gates (import-linter contracts, ruff C901 complexity) that prevent recurrence of the issues found in phases 24-26 (completed 2026-03-07)

## Phase Details

### Phase 24: Audit Baseline
**Goal**: Operators have a complete, evidence-based understanding of codebase health before any remediation work begins
**Depends on**: Nothing (first phase of v1.5)
**Requirements**: AUDT-01, AUDT-02, AUDT-03, AUDT-04, AUDT-05
**Success Criteria** (what must be TRUE):
  1. Every test file in the test suite is classified as behavioral, integration, structural-conformance, or milestone-marker, and a coverage gap report identifies which pipeline stages lack behavioral tests
  2. Running `vulture` at 80% confidence produces a triaged findings list with a false-positive whitelist so the operator can distinguish real dead code from framework callbacks
  3. A branch coverage report from pytest-cov identifies the lowest-coverage modules, giving a ranked list of where behavioral tests are most needed
  4. A radon complexity audit identifies all functions exceeding cyclomatic complexity 10, producing a remediation candidate list for phases 25-26
  5. The E2E pipeline smoke test (ingest -> features -> train -> batch -> monitor) has been executed and a log documents which sequential steps pass and which fail, with specific error messages for failures
**Plans**: 3 plans

Plans:
- [ ] 24-01-PLAN.md — Classify test suite, add pytest markers, delete milestone-markers, produce coverage gap matrix
- [ ] 24-02-PLAN.md — Run vulture dead code scan with whitelist triage and radon complexity audit
- [ ] 24-03-PLAN.md — Run branch coverage report, E2E smoke test, and synthesize AUDIT-REPORT.md

### Phase 25: Critical Correctness Remediation
**Goal**: All silently broken production code paths are fixed with one-fix-one-test discipline, and missing correctness guardrails are created
**Depends on**: Phase 24 (audit baseline identifies and prioritizes issues)
**Requirements**: CORR-01, CORR-02, CORR-03, CORR-04, CORR-05, CORR-06, LEAK-01, LEAK-02
**Success Criteria** (what must be TRUE):
  1. The autonomous retrainer subprocess command no longer passes `--tau` to the CLI, and a test asserts the subprocess argument list matches the actual CLI interface so this class of bug is caught automatically
  2. Walk-forward CV metric writing and reading use the same key names, and a test verifies round-trip consistency (write then read returns the same scores)
  3. `regression_metrics()` returns computed values without performing file I/O as a side effect, and a test confirms no files are written when the function is called
  4. All `assert isinstance` statements in production code paths (non-test files) are replaced with explicit `if not isinstance: raise TypeError` guards that survive `python -O`
  5. `tests/features/test_leakage.py` exists and contains a PR-AUC guardrail test as documented in CLAUDE.md, failing if train/test leakage is detected
  6. API route freq/horizon defaults are sourced from default.yaml rather than hardcoded, and a test confirms the API defaults match the config file values
  7. OBV fold-boundary leakage impact is empirically assessed with a walk-forward comparison (with vs without OBV), and if confirmed material, the cumsum is computed fold-aware
**Plans**: 4 plans

Plans:
- [ ] 25-01-PLAN.md — Fix retrainer CLI contract (--tau removal) and CV metric key naming consistency
- [ ] 25-02-PLAN.md — Separate regression_metrics I/O from computation, replace assert isinstance with TypeError guards
- [ ] 25-03-PLAN.md — Create test_leakage.py PR-AUC guardrail, align API route defaults with config
- [ ] 25-04-PLAN.md — Assess OBV fold-boundary leakage empirically and implement fold-aware cumsum

### Phase 26: Architecture Targeted Fixes
**Goal**: The highest-friction architecture violations are repaired so that future development does not re-encounter duplicated logic, broken test isolation, or cross-layer imports
**Depends on**: Phase 25 (correctness must be restored before architecture contracts are defined against it)
**Requirements**: ARCH-01, ARCH-02, ARCH-03, ARCH-04
**Success Criteria** (what must be TRUE):
  1. `_generate_price_features` and `_join_auxiliary_features` are accessible as public functions with stable interfaces, and the 3 external callers (cli.py, autonomous/predictor.py, autonomous/continuous_trainer.py) import from the public API instead of reaching into private internals
  2. Price loading logic exists in a single shared function, and the 3 divergent implementations in cli.py, autonomous/predictor.py, and autonomous/continuous_trainer.py are replaced with calls to it
  3. A `reset_runtime_config()` function exists in config/loader.py, test fixtures use it, and running tests in any order produces the same results (no config state leaks between tests)
  4. `api/routes/system.py` does not import from `gui/` — the cross-layer dependency is eliminated and shared utilities live in an appropriate lower layer
**Plans**: 2 plans

Plans:
- [x] 26-01-PLAN.md — Promote private feature pipeline functions to public API and consolidate price loading into shared module
- [x] 26-02-PLAN.md — Add config reset for test isolation and eliminate API-to-GUI cross-layer imports

### Phase 27: Verification & Guardrail Hardening
**Goal**: CI gates prevent recurrence of the architecture drift and complexity creep found during this audit
**Depends on**: Phase 26 (contracts must be written against the corrected architecture, not the pre-fix state)
**Requirements**: ARCH-05, ARCH-06
**Success Criteria** (what must be TRUE):
  1. import-linter contracts are committed and enforced in CI, and introducing a cross-layer import (e.g., api importing from gui) causes CI to fail
  2. ruff C901 complexity gate is active in CI with max-complexity = 10, and adding a function with cyclomatic complexity > 10 causes CI to fail
**Plans**: 1 plan

Plans:
- [ ] 27-01-PLAN.md — Enable ruff C901 complexity gate, install import-linter forbidden contract, and wire both into CI

### Phase 28: Activate Fold-Aware OBV in Production Pipeline
**Goal**: Wire fold_boundaries from walk-forward CV splits into generate_price_features() so obv_fold_aware() is used in production retraining and closes the LEAK-02 production activation gap
**Depends on**: Phase 25 (obv_fold_aware() implemented), Phase 26 (public API)
**Requirements**: LEAK-02 (production activation)
**Success Criteria** (what must be TRUE):
  1. `generate_price_features()` accepts an optional `fold_boundaries` parameter and passes it through to `obv_fold_aware()`
  2. The walk-forward CV loop (or the retraining path that calls it) extracts fold split boundaries and supplies them to `generate_price_features()`
  3. A test confirms that when `fold_boundaries` is provided, `obv_fold_aware()` is called instead of the cumulative OBV path
  4. Existing tests pass unchanged (no regressions)
**Plans**: 1 plan

Plans:
- [ ] 28-01-PLAN.md — Wire fold_boundaries into generate_price_features and activate obv_fold_aware in CV/retraining path

## Progress

**Execution Order:**
Phases execute in numeric order: 24 -> 25 -> 26 -> 27 -> 28

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 24. Audit Baseline | 3/3 | Complete    | 2026-03-04 | - |
| 25. Critical Correctness Remediation | 4/4 | Complete   | 2026-03-06 | - |
| 26. Architecture Targeted Fixes | 2/2 | Complete    | 2026-03-07 | - |
| 27. Verification & Guardrail Hardening | 1/1 | Complete    | 2026-03-07 | - |
| 28. Activate Fold-Aware OBV | 1/1 | Complete   | 2026-03-08 | - |
