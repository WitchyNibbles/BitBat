---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: Codebase Health Audit & Critical Remediation
status: unknown
last_updated: "2026-03-08T03:09:14.810Z"
progress:
  total_phases: 24
  completed_phases: 24
  total_plans: 63
  completed_plans: 63
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-04)

**Core value:** A reliable prediction system where operators can trust that monitoring outputs correspond to real, active prediction flows for the configured runtime pair.
**Current focus:** Phase 28 — Activate Fold-Aware OBV

## Current Position

Phase: 28 of 28 (Activate Fold-Aware OBV)
Plan: 1 of 1 in Phase 28 -- COMPLETE
Status: Phase 28 Plan 01 complete
Last activity: 2026-03-08 - Completed 28-01: wire fold_boundaries into ContinuousTrainer._do_retrain()

Progress: [##########] 1/1 plans completed (Phase 28 Plan 01)

## Performance Metrics

**Velocity:**
- Total plans completed: 9
- Average duration: 5min
- Total execution time: 0.88 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 24 | 3/3 | 20min | 7min |
| 25 | 4/4 | 17min | 4min |
| 26 | 2/2 | 14min | 7min |
| 27 | 1/1 | 8min | 8min |

## Accumulated Context

### Decisions Summary

- v1.5 is a comprehensive audit milestone: find all issues, fix critical, catalog the rest.
- Correctness before architecture: phases 24-25 establish baseline and fix breaks before phases 26-27 touch structure.
- OBV leakage (LEAK-01/02) scoped as assess-then-fix: empirical comparison first, fold-aware fix conditional on results.
- Style-only fixes explicitly out of scope to avoid audit noise.
- v1.5 phases start at 24 (continuing from v1.4 phases 20-23).
- Module-level pytestmark used for all test files (behavioral/integration/structural taxonomy).
- 16 *_complete.py files retained (exercise real production code); 1 deleted (pure source-reader).
- All 14 v1.5 requirements confirmed as coverage gaps (expected: requirements target known issues).
- Vulture: zero genuine dead code at 80% confidence; all 17 findings are pytest fixture false positives.
- CLI monolith complexity (5 functions CC 15-37) mapped to DEBT-01, deferred to v1.6+.
- LivePredictor.predict_latest (CC=28) identified as highest actionable complexity target for v1.5.
- CORR-02 downgraded from CRITICAL to HIGH: primary key lookup works correctly, risk is naming confusion and fragile cascade.
- CORR-01 location corrected: --tau passed to features build (not model train as research stated).
- Smoke test used real yfinance data; model output path and autonomous.db path don't fully respect data_dir (DEBT-02/DEBT-03).
- AUDIT-REPORT.md synthesizes 26 findings with severity triage, ready to drive phases 25-27 planning.
- Writer key renamed from average_balanced_accuracy to mean_directional_accuracy; reader cascade kept for backward compat with old cv_summary.json files.
- CLI contract test uses Click CliRunner --help parsing to extract valid options dynamically (self-updating guard).
- API route defaults sourced from config via api/defaults.py helper; module-level constants computed once at import to avoid per-request YAML parsing.
- PR-AUC guardrail threshold set at 0.7 (random labels yield ~0.5; 0.7 catches genuine leakage with margin).
- OBV fold-boundary leakage empirically NOT material (2.33pp < 3pp threshold); fold-aware fix implemented as correct practice regardless.
- ARCH-01/02: Backward-compat aliases (_generate_price_features = generate_price_features) kept in build.py; two load_prices variants created (glob-based for autonomous pipeline, flat-file for CLI); AST structural guard added to tests.
- ARCH-03/04: Preset dataclass and get_ingestion_status relocated to bitbat.common layer; gui modules re-export for backward compat; reset_runtime_config() added to config/loader.py for clean test teardown; AST structural guards block future api->gui imports.
- ARCH-05/06: C901 max-complexity=10 gate active in ruff; import-linter forbidden contract blocks api->gui imports; both enforced in CI lint job. Used forbidden (not layers) contract type to avoid over-constraining cross-cutting modules. orchestrator.py transitive gui import fixed (missed in phase 26).
- LEAK-02 closed: fold_boundaries=[self.train_window_bars] passed to generate_price_features() in ContinuousTrainer._do_retrain(); inference paths (predictor.py, batch cli) intentionally retain fold_boundaries=None.

### Pending Todos

(None)

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 1 | fix metrics.py to use _FREQ/_HORIZON from api/defaults.py | 2026-03-07 | c03721f | [1-fix-metrics-py-to-use-freq-horizon-from-](.planning/quick/1-fix-metrics-py-to-use-freq-horizon-from-/) |

### Blockers/Concerns

- Preserve all v1.0-v1.4 validated contracts as non-regression constraints.
- Audit fixes must not break existing `make test-release` gates.
- OBV fold-boundary fix scope may expand if WalkForwardValidator integration is needed (research flag from SUMMARY.md).

## Session Continuity

Last session: 2026-03-08
Stopped at: Completed 28-01-PLAN.md (wire fold_boundaries into ContinuousTrainer._do_retrain())
Resume with: Phase 28 Plan 01 complete — LEAK-02 production activation gap closed
