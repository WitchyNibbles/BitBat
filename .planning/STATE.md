---
gsd_state_version: 1.0
milestone: v1.6
milestone_name: Accuracy Recovery & Technical Debt Remediation
status: completed
stopped_at: Completed 32-02-PLAN.md
last_updated: "2026-03-12T09:25:16.592Z"
last_activity: "2026-03-12 — 32-01 complete: cli.py → cli/ package skeleton; _helpers.py with 21 private helpers; 41 tests pass; lint-imports 1/0"
progress:
  total_phases: 7
  completed_phases: 3
  total_plans: 9
  completed_plans: 8
  percent: 97
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-08)

**Core value:** A reliable prediction system where operators can trust that monitoring outputs correspond to real, active prediction flows for the configured runtime pair.
**Current focus:** Milestone v1.6 — Phase 30: Fix and Reset

## Current Position

Phase: 32 of 35 (CLI Decomposition — Plan 02 COMPLETE)
Plan: 32-02 complete; 32-03 next
Status: Phase 32 Plan 02 Complete — 8 simple command groups extracted to commands/ submodules; __init__.py -662 lines; all groups wired via add_command()
Last activity: 2026-03-12 — 32-02 complete: 8 command group modules in commands/; __init__.py reduced to model+monitor+wiring; bitbat --help all 10 groups present

Progress: [██████████] 97% (59/61 plans complete)

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
| Phase 31-accuracy-guardrail P01 | 3 | 2 tasks | 3 files |
| Phase 32-cli-decomposition P01 | 20min | 3 tasks | 4 files |
| Phase 32-cli-decomposition P02 | 16 | 3 tasks | 9 files |

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
- 31-01: check_accuracy_guardrail() is module-level function (not a method) to enable unit testing without model artifact on disk.
- 31-01: Mock patch target is bitbat.autonomous.agent.send_alert (from-import binding location, not the alerting module).
- 31-01: Default threshold 0.40, min_predictions_required=10 prevents warmup false positives after operator system reset.
- 32-01: noqa: F401 on re-exported domain symbols in bitbat.cli.__init__ — required for monkeypatch compatibility (tests patch bitbat.cli.xgb, bitbat.cli.walk_forward, etc.).
- 32-01: cli.py deleted only after cli/__init__.py fully written — Python cannot coexist both cli.py and cli/ at same import path.
- 32-02: run_strategy, summarize_backtest kept as bitbat.cli re-exports from original module-level imports; monkeypatch targets updated in Plan 03.
- 32-02: features_build monkeypatch failures (bitbat.cli.build_xy) follow same pattern as batch — deferred to Plan 03.
- 32-02: 4 test failures are all monkeypatch targeting (tests patch bitbat.cli.* but commands hold direct refs in command modules); expected per plan.

### Pending Todos

(None)

### Blockers/Concerns

- Phase 30 fix scope determined: 3 bugs in Stage 4 (reg:squarederror), Stage 5 (sign-only inference), Stage 6 (tau=0.0 hardcode + price gaps).
- Accuracy recovery (phases 29-31) and tech debt (phases 32-35) can execute in parallel tracks if needed.
- Preserve autonomous.db backward compatibility throughout DEBT-03 unification.
- After Phase 30 fixes, tests/diagnosis/ assertions must be inverted (from "bug exists" to "bug fixed"). DONE in 30-02.
- Operator must run `bitbat system reset --yes` before retraining to clear pre-fix autonomous.db predictions.
- Phase 31 complete. Phase 32 (Tech Debt) is next. Operator still needs `bitbat system reset --yes` + retrain to clear pre-fix autonomous.db and activate the live accuracy guardrail.

## Session Continuity

Last session: 2026-03-12T09:25:16.589Z
Stopped at: Completed 32-02-PLAN.md
Resume with: Plan Phase 29 (Diagnosis) — investigate live accuracy collapse
