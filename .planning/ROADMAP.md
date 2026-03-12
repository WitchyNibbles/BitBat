# Roadmap: BitBat Reliability and Timeline Evolution

## Milestones

- ✅ **v1.0 Reliability and Timeline Evolution** — Phases 1-9 shipped on 2026-02-25 ([roadmap archive](milestones/v1.0-ROADMAP.md), [requirements archive](milestones/v1.0-REQUIREMENTS.md), [audit archive](milestones/v1.0-MILESTONE-AUDIT.md)).
- ✅ **v1.1 UI-First Simplification** — Phases 10-12 shipped on 2026-02-25 ([roadmap archive](milestones/v1.1-ROADMAP.md), [requirements archive](milestones/v1.1-REQUIREMENTS.md), [audit archive](milestones/v1.1-MILESTONE-AUDIT.md)).
- ✅ **v1.2 BTC Prediction Accuracy Evolution** — Phases 13-16 shipped on 2026-02-26 ([roadmap archive](milestones/v1.2-ROADMAP.md), [requirements archive](milestones/v1.2-REQUIREMENTS.md)).
- ✅ **v1.3 Autonomous Monitor Alignment and Metrics Integrity** — Phases 17-19 shipped on 2026-02-26 ([roadmap archive](milestones/v1.3-ROADMAP.md), [requirements archive](milestones/v1.3-REQUIREMENTS.md)).
- ✅ **v1.4 Configuration Alignment** — Phases 20-23 shipped on 2026-03-01 ([roadmap archive](milestones/v1.4-ROADMAP.md), [requirements archive](milestones/v1.4-REQUIREMENTS.md), [audit archive](milestones/v1.4-MILESTONE-AUDIT.md)).
- ✅ **v1.5 Codebase Health Audit & Critical Remediation** — Phases 24-28 shipped on 2026-03-08 ([roadmap archive](milestones/v1.5-ROADMAP.md), [requirements archive](milestones/v1.5-REQUIREMENTS.md), [audit archive](milestones/v1.5-MILESTONE-AUDIT.md)).

## Phases

<details>
<summary>✅ v1.5 Codebase Health Audit & Critical Remediation (Phases 24-28) — SHIPPED 2026-03-08</summary>

- [x] **Phase 24: Audit Baseline** — 3/3 plans — completed 2026-03-04
- [x] **Phase 25: Critical Correctness Remediation** — 4/4 plans — completed 2026-03-06
- [x] **Phase 26: Architecture Targeted Fixes** — 2/2 plans — completed 2026-03-07
- [x] **Phase 27: Verification & Guardrail Hardening** — 1/1 plan — completed 2026-03-07
- [x] **Phase 28: Activate Fold-Aware OBV** — 1/1 plan — completed 2026-03-08

</details>

### v1.6 Accuracy Recovery & Technical Debt Remediation (Gap Closure In Progress)

**Milestone Goal:** Diagnose and fix the live prediction accuracy collapse (~1%), restore the pipeline to a working state with clean reset + retrain, add an accuracy collapse guardrail to the monitor, and eliminate the four deferred tech debt items.

- [x] **Phase 29: Diagnosis** — Identify and document the root cause of live accuracy collapse before any fix is applied (completed 2026-03-08)
- [x] **Phase 30: Fix & Reset** — Fix root cause in code, provide a clean reset + retrain procedure, and verify accuracy exceeds random baseline (completed 2026-03-08)
- [x] **Phase 31: Accuracy Guardrail** — Add monitor alert that fires when realized accuracy falls below a configurable threshold (completed 2026-03-08)
- [x] **Phase 32: CLI Decomposition** — Split cli.py monolith (1802+ lines, 53 functions) into focused command modules (completed 2026-03-12)
- [x] **Phase 33: Path Centralization** — Replace all 15+ hardcoded Path("models")/Path("metrics") sites with config-driven path resolution (completed 2026-03-12)
- [x] **Phase 34: DB Unification** — Consolidate dual DB access (SQLAlchemy ORM + raw sqlite3) into a single consistent approach (completed 2026-03-12)
- [x] **Phase 35: XGBoost Fix** — Replace reg:squarederror with a classification objective and retrain the model (completed 2026-03-12)
- [ ] **Phase 36: Live Recovery Evidence Closure** — Capture formal post-reset/post-retrain evidence that the recovered pipeline exceeds the realized-accuracy baseline and closes FIXR-03
- [ ] **Phase 37: CLI Decomposition Re-Verification** — Reconcile the stale Phase 32 verification artifact and close DEBT-01 with a passed verification record

## Phase Details

### Phase 29: Diagnosis
**Goal**: Operators can identify which pipeline stage caused the live accuracy collapse and have a documented, reproducible trace before any fix is applied
**Depends on**: Phase 28
**Requirements**: DIAG-01, DIAG-02
**Success Criteria** (what must be TRUE):
  1. Operator can run a CLI command or test sequence that surfaces which stage (ingestion, features, labels, model, serving) produced incorrect predictions
  2. A written root-cause document exists that identifies the specific bug, includes a reproducible repro trace, and was committed before any fix code was merged
  3. The diagnosed stage is confirmed by comparing pipeline outputs at each boundary (raw data → feature values → label distribution → model probabilities → served predictions)
**Plans**: 2 plans
Plans:
- [ ] 29-01-PLAN.md — Create diagnostic test scaffold (pipeline stage trace + ROOT_CAUSE.md structural tests)
- [ ] 29-02-PLAN.md — Write ROOT_CAUSE.md and confirm full diagnostic suite passes

### Phase 30: Fix & Reset
**Goal**: The root cause of live accuracy ~1% is fixed in code, a clean reset procedure is available via CLI, and a retrained model achieves directional accuracy above random baseline
**Depends on**: Phase 29
**Requirements**: FIXR-01, FIXR-02, FIXR-03
**Success Criteria** (what must be TRUE):
  1. The specific code defect identified in Phase 29 is corrected and covered by a deterministic automated test that would have caught the regression
  2. Operator can run documented CLI command(s) to delete data/, models/, and autonomous.db and reach a clean-slate state without manual file manipulation
  3. After reset + retrain, realized directional accuracy on predictions that have passed the horizon exceeds 33% (random baseline for 3-class up/down/flat)
  4. The fix and reset procedure are verified by automated tests that run as part of the existing test suite
**Plans**: 3 plans
Plans:
- [ ] 30-01-PLAN.md — Fix three root-cause bugs (train.py, infer.py, validator.py) + unit tests
- [ ] 30-02-PLAN.md — Invert Phase 29 diagnostic tests + add bitbat system reset CLI command
- [ ] 30-03-PLAN.md — Accuracy gate verification + operator reset + retrain checkpoint

### Phase 31: Accuracy Guardrail
**Goal**: The monitor agent alerts operators when realized directional accuracy falls below a configurable threshold, preventing silent accuracy collapse from going undetected
**Depends on**: Phase 30
**Requirements**: FIXR-04
**Success Criteria** (what must be TRUE):
  1. Monitor agent emits a structured alert when realized accuracy on the rolling window of realized predictions drops below the configured threshold (default: 40%)
  2. The accuracy threshold is configurable via the existing config YAML without code changes
  3. The guardrail fires under simulated low-accuracy conditions in an automated test (not only in live operation)
  4. Alert includes the observed accuracy value, the threshold, and the number of realized predictions used in the calculation
**Plans**: 1 plan
Plans:
- [ ] 31-01-PLAN.md — Add accuracy guardrail config + check_accuracy_guardrail function + agent wiring + 5 behavioral tests

### Phase 32: CLI Decomposition
**Goal**: cli.py is split into focused command-group modules with no behavioral change, eliminating the 1802+ line monolith and satisfying the deferred DEBT-01 obligation
**Depends on**: Phase 28
**Requirements**: DEBT-01
**Success Criteria** (what must be TRUE):
  1. cli.py is reduced to a thin entry point (registration only); all 9 command groups live in dedicated sub-modules
  2. Every existing CLI command (`bitbat --help` surface) continues to work identically — no command names, flags, or output formats changed
  3. The ruff C901 complexity gate passes on all new modules without noqa suppressions
  4. Existing CLI tests pass without modification
**Plans**: 3 plans
Plans:
- [ ] 32-01-PLAN.md — Create cli/ package skeleton: _helpers.py + __init__.py + commands/__init__.py
- [ ] 32-02-PLAN.md — Move 8 simple command groups (prices, news, features, backtest, batch, validate, ingest, system) into commands/
- [ ] 32-03-PLAN.md — Move model + monitor groups, refactor model_cv for C901, update test monkeypatch targets

### Phase 33: Path Centralization
**Goal**: All 15+ hardcoded Path("models") and Path("metrics") occurrences are replaced with a single config-driven resolution point, so operators can relocate artifacts by changing one config value
**Depends on**: Phase 28
**Requirements**: DEBT-02
**Success Criteria** (what must be TRUE):
  1. A single canonical path-resolution helper exists (e.g., config.paths.models_dir, config.paths.metrics_dir) and all modules use it
  2. No remaining literal Path("models") or Path("metrics") strings exist in src/ (verified by a structural grep test or linter rule)
  3. Changing the paths in config YAML redirects all artifact reads and writes without code changes
  4. Existing tests pass and the smoke test still produces artifacts in the expected location
**Plans**: 2 plans
Plans:
- [ ] 33-01-PLAN.md — Add resolve_models_dir / resolve_metrics_dir helpers to config loader + TDD test scaffold
- [ ] 33-02-PLAN.md — Sweep all 16 hardcoded path literals across 10 src files; turn structural grep tests GREEN

### Phase 34: DB Unification
**Goal**: All database access uses a single consistent approach — eliminating the split between SQLAlchemy ORM and raw sqlite3 — so the codebase has one query pattern, one connection lifecycle, and one schema migration path
**Depends on**: Phase 28
**Requirements**: DEBT-03
**Success Criteria** (what must be TRUE):
  1. A single DB access layer is chosen and all modules use it; the other approach has zero remaining call sites in src/
  2. The autonomous.db schema is preserved — no data migration required for existing databases
  3. All autonomous monitor tests pass; no behavioral change is observable from the CLI or API
  4. Connection lifecycle (open, close, error handling) is consistent across all DB call sites
**Plans**: 3 plans
Plans:
- [x] 34-01-PLAN.md — Establish AutonomousDB read facade, transient-lock retry/circuit-breaker, and atomic write helpers
- [x] 34-02-PLAN.md — Migrate `/system` API routes to AutonomousDB and add fail-fast API coverage
- [x] 34-03-PLAN.md — Migrate Streamlit DB helpers, unify retraining transactions, and add a structural no-sqlite3 gate

### Phase 35: XGBoost Fix
**Goal**: The XGBoost model uses a classification objective (multi:softprob) instead of the regression objective (reg:squarederror), eliminating the mismatch between training objective and prediction use case
**Depends on**: Phase 30
**Requirements**: DEBT-04
**Success Criteria** (what must be TRUE):
  1. The XGBoost objective is set to multi:softprob (or equivalent) in training config and confirmed by inspecting the saved model artifact
  2. Model outputs are valid class probabilities (sum to 1.0 per row, values in [0,1]) verified by an automated test
  3. Existing model training, inference, and evaluation tests pass with the new objective
  4. After retrain with the corrected objective, walk-forward CV PR-AUC meets the existing 0.7 guardrail threshold
**Plans**: 2 plans
Plans:
- [x] 35-01-PLAN.md — Make walk-forward and optimizer classification-aware for label targets while preserving regression fallback
- [x] 35-02-PLAN.md — Wire CLI CV/optimization to labels and surface PR-AUC-aware summaries

### Phase 36: Live Recovery Evidence Closure
**Goal**: The operator reset + retrain path is verified end-to-end on fresh runtime data, and the recovered system records realized directional accuracy above the random baseline with formal saved verification evidence
**Depends on**: Phase 35
**Requirements**: FIXR-03
**Gap Closure**: Closes the Phase 30 verification deferral, the 30→31 integration gap, and the broken operator recovery flow identified in the v1.6 milestone audit
**Success Criteria** (what must be TRUE):
  1. A documented reset + retrain procedure is executed against fresh runtime state, not stale pre-fix `autonomous.db` data
  2. Saved verification evidence shows realized directional accuracy above 33% on fresh post-reset predictions
  3. Phase 30 or the new closure phase has a passed verification artifact that formally closes the deferred FIXR-03 checkpoint
  4. The operator recovery flow (`system reset` → retrain → realized outcomes → diagnosis accuracy check) is reproducible from the repo documentation
**Plans**: 2 plans
Plans:
- [ ] 36-01-PLAN.md — Add a recovery-evidence harness, config-aware diagnosis resolution, and docs/contracts for the reset + retrain flow
- [ ] 36-02-PLAN.md — Run the sandbox reset + retrain evidence flow, update Phase 30 verification, and record passed FIXR-03 evidence

### Phase 37: CLI Decomposition Re-Verification
**Goal**: The CLI decomposition is formally re-verified so the saved verification evidence matches the now-green code/test surface and DEBT-01 is audit-clean
**Depends on**: Phase 36
**Requirements**: DEBT-01
**Gap Closure**: Closes the stale Phase 32 verification gap and the broken milestone archive flow identified in the v1.6 milestone audit
**Success Criteria** (what must be TRUE):
  1. The original Phase 32 monkeypatch-target failure is rechecked against the current codebase and fixed if still present
  2. CLI regression evidence is rerun and saved in a passed verification artifact
  3. DEBT-01 is supported by matching SUMMARY, VERIFICATION, and REQUIREMENTS traceability records
  4. Milestone archive evidence no longer depends on inference from later phases to claim CLI decomposition is complete
**Plans**: TBD

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 24. Audit Baseline | v1.5 | 3/3 | Complete | 2026-03-04 |
| 25. Critical Correctness Remediation | v1.5 | 4/4 | Complete | 2026-03-06 |
| 26. Architecture Targeted Fixes | v1.5 | 2/2 | Complete | 2026-03-07 |
| 27. Verification & Guardrail Hardening | v1.5 | 1/1 | Complete | 2026-03-07 |
| 28. Activate Fold-Aware OBV | v1.5 | 1/1 | Complete | 2026-03-08 |
| 29. Diagnosis | v1.6 | 2/2 | Complete | 2026-03-08 |
| 30. Fix & Reset | 3/3 | Complete   | 2026-03-08 | - |
| 31. Accuracy Guardrail | 1/1 | Complete    | 2026-03-08 | - |
| 32. CLI Decomposition | 3/3 | Complete    | 2026-03-12 | - |
| 33. Path Centralization | 2/2 | Complete   | 2026-03-12 | - |
| 34. DB Unification | 3/3 | Complete    | 2026-03-12 | - |
| 35. XGBoost Fix | v1.6 | 2/2 | Complete | 2026-03-12 |
| 36. Live Recovery Evidence Closure | v1.6 | 0/2 | Planned | - |
| 37. CLI Decomposition Re-Verification | v1.6 | 0/TBD | Not started | - |
