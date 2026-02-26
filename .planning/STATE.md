# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-25)

**Core value:** A reliable prediction system where operators can trust that monitoring runs without DB failures and the timeline shows clear prediction vs. outcome history.
**Current focus:** Phase 16 planning for promotion guardrails and optimization safety

## Current Position

Milestone: v1.2 (BTC Prediction Accuracy Evolution)
Phase: 16-promotion-guardrails-and-optimization-safety (not started)
Status: Phase 15 executed and verified; ready to plan/execute Phase 16
Last activity: 2026-02-26 - Completed Phase 15 for EVAL-01/EVAL-02/EVAL-03

Progress: [███████░░░] 75% for v1.2 (3/4 phases complete)

## Milestone Metrics

- Planned phases: 4 (13-16)
- Planned requirements: 11
- Plans: 9 complete (Phases 13-15)
- Tasks: 27 complete (plans 13-01..03, 14-01..03, 15-01..03)
- Source context: `deep-research-report.md`

## Accumulated Context

### Decisions Summary

- Streamlit runtime surface is intentionally constrained to five supported operator views.
- Legacy Backtest/Pipeline routes are retirement-guarded with supported-page guidance.
- Home prediction rendering now tolerates missing confidence/optional fields without crashes.
- `make test-release` remains the canonical acceptance command for D1/D2/D3 gates.
- Simplified UI behavior is release-wired by dedicated phase-level regression and smoke suites.
- v1.2 prioritizes leakage control, retraining cadence, walk-forward rigor, and cost-aware model selection.
- Tree-ensemble baselines are the first accuracy benchmark before considering transformer or graph model expansion.
- As-of alignment with no-future-match enforcement is now a dataset-level invariant (DATA-01).
- Return targets (`r_forward`) and direction labels now share one horizon-aware labeling path (DATA-02).
- Triple-barrier event labels are available as an optional dataset/CLI mode while default return-direction behavior is unchanged (LABL-01).
- Phase 14 shipped dual baseline family support (`xgb`, `random_forest`) with comparable persistence artifacts (MODL-01).
- Retraining and CV now use explicit rolling train/backtest windows configurable through CLI/config and autonomous flows (MODL-02).
- Per-window regime/drift diagnostics are emitted in evaluation, retraining artifacts, and monitor CLI output (MODL-03).
- Walk-forward evaluation now enforces explicit purge/embargo leakage controls from CLI/config (EVAL-01).
- Backtest and walk-forward metrics now include explicit fee/slippage attribution and net-vs-gross reporting (EVAL-02).
- Candidate reports and champion decisions are persisted, deterministic, and enforced in retrainer deployment gates (EVAL-03).

### Pending Todos

- Plan Phase 16 (`$gsd-plan-phase 16`) for EVAL-04 and OPER-02 guardrail requirements.
- Execute Phase 16 and verify nested-validation safety and promotion gate behavior.
- Keep autonomous promotion tied to incumbent-beating out-of-sample and drawdown-safe decisions.

### Blockers/Concerns

- No active blockers. Main risk remains overfitting during optimization/promotion workflows in Phase 16.

## Session Continuity

Last session: 2026-02-26
Stopped at: Phase 15 complete and verified
Resume with: `$gsd-plan-phase 16`
