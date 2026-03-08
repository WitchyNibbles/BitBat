---
gsd_state_version: 1.0
milestone: v1.6
milestone_name: Accuracy Recovery & Technical Debt Remediation
status: completed
stopped_at: Completed 30-03-PLAN.md
last_updated: "2026-03-08T17:09:29.755Z"
last_activity: "2026-03-08 — 30-02 complete: diagnostic tests inverted, bitbat system reset --yes added, 649 tests pass"
progress:
  total_phases: 7
  completed_phases: 2
  total_plans: 5
  completed_plans: 5
  percent: 98
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-08)

**Core value:** A reliable prediction system where operators can trust that monitoring outputs correspond to real, active prediction flows for the configured runtime pair.
**Current focus:** Milestone v1.6 — Phase 30: Fix and Reset

## Current Position

Phase: 30 of 35 (Fix and Reset — COMPLETE)
Plan: 03 complete; Phase 31 (Accuracy Guardrail) next
Status: Phase 30 Plan 03 Complete — accuracy verification checkpoint acknowledged; FIXR-03 deferred to Phase 31 (unit tests confirm fixes, live accuracy requires operator reset + retrain)
Last activity: 2026-03-08 — 30-03 complete: accuracy gate test verified (fails on pre-fix DB as expected), reset CLI tests 3/3 pass, checkpoint auto-approved, Phase 30 complete

Progress: [██████████] 100% (57/57 plans complete)

## Performance Metrics

**Velocity (v1.5 baseline):**
- Total plans completed: 11
- Average duration: 5min
- Total execution time: ~1 hour

**By Phase (v1.5):**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 24 | 3/3 | 20min | 7min |
| 25 | 4/4 | 17min | 4min |
| 26 | 2/2 | 14min | 7min |
| 27 | 1/1 | 8min | 8min |
| 28 | 1/1 | — | — |
| Phase 29-diagnosis P01 | 1 | 2 tasks | 3 files |
| Phase 29-diagnosis P02 | 5 | 2 tasks | 1 files |
| Phase 30-fix-and-reset P01 | 10 | 3 tasks | 8 files |
| Phase 30-fix-and-reset P02 | 8 | 2 tasks | 3 files |
| Phase 30-fix-and-reset P03 | 6min | 2 tasks | 0 files |

## Accumulated Context

### Decisions Summary

- 29-01: Diagnosis-first TDD — tests assert bugs are present before fix code is written; assertions inverted after Phase 30.
- 29-01: test_root_cause_md_exists intentionally RED (FAIL) — gates Phase 30 fix code on ROOT_CAUSE.md being committed first.
- 29-02: ROOT_CAUSE.md committed before Phase 30 fix code — DIAG-02 constraint satisfied; structural test now GREEN.
- v1.6 diagnosis-first: Phase 29 must document root cause before Phase 30 applies any fix.
- Phase 35 (XGBoost Fix) depends on Phase 30 (reset + retrain) to validate the corrected objective end-to-end.
- Phases 32-34 (tech debt) depend only on Phase 28 and are independent of the accuracy recovery track (29-31).
- DEBT-01: CLI monolith — 11 noqa:C901 suppressions on pre-existing complexity; new modules must pass without suppressions.
- DEBT-02: Path hardcoding — smoke test confirmed model output path and autonomous.db path don't fully respect data_dir.
- DEBT-03: DB unification — autonomous.db schema must be preserved; no data migration required.
- DEBT-04: XGBoost objective — reg:squarederror is a regression objective; multi:softprob is the correct 3-class classification objective.
- All v1.0-v1.5 validated contracts are non-regression constraints for every v1.6 phase.
- 30-01: DIRECTION_CLASSES identical in train.py and infer.py, guarded by test_direction_classes_consistent_across_modules.
- 30-01: predict_bar returns predicted_return=None and p_flat for 3-class classification; directional_confidence kept but not called.
- 30-01: cli model_train uses require_label=True so label column survives ensure_feature_contract column filtering.
- 30-01: validator tau wired from constructor/config; get_runtime_config() fallback with default 0.01.
- 30-02: Diagnostic tests skip when DB/model absent; pre-fix DB data causes failure — expected until operator runs system reset.
- 30-02: system reset deletes data/ and models/ via shutil.rmtree; autonomous.db deleted separately only if outside data_dir.
- 30-02: Path.is_relative_to() used (Python 3.11) for autonomous.db containment check.
- 30-03: FIXR-03 deferred — live accuracy verification requires operator to run bitbat system reset --yes and retrain; unit tests confirm code correctness; live verification deferred to Phase 31.
- 30-03: Diagnostic DB tests correctly fail on pre-fix autonomous.db (38/266 = 14.3%) — expected pre-reset state, not a code bug; operator must run reset before tests pass.

### Pending Todos

(None)

### Blockers/Concerns

- Phase 30 fix scope determined: 3 bugs in Stage 4 (reg:squarederror), Stage 5 (sign-only inference), Stage 6 (tau=0.0 hardcode + price gaps).
- Accuracy recovery (phases 29-31) and tech debt (phases 32-35) can execute in parallel tracks if needed.
- Preserve autonomous.db backward compatibility throughout DEBT-03 unification.
- After Phase 30 fixes, tests/diagnosis/ assertions must be inverted (from "bug exists" to "bug fixed"). DONE in 30-02.
- Operator must run `bitbat system reset --yes` before retraining to clear pre-fix autonomous.db predictions.
- Phase 30 complete. Phase 31 (Accuracy Guardrail) is unblocked. Operator should run system reset at start of Phase 31.

## Session Continuity

Last session: 2026-03-08T17:09:29.752Z
Stopped at: Completed 30-03-PLAN.md
Resume with: Plan Phase 29 (Diagnosis) — investigate live accuracy collapse
