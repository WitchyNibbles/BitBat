# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-25)

**Core value:** A reliable prediction system where operators can trust that monitoring runs without DB failures and the timeline shows clear prediction vs. outcome history.
**Current focus:** Phase 15 plans are drafted and ready for execution

## Current Position

Milestone: v1.2 (BTC Prediction Accuracy Evolution)
Phase: 15-cost-aware-walk-forward-evaluation (planned)
Status: Phase 15 plans created and checker-verified; ready for execution
Last activity: 2026-02-26 - Planned Phase 15 for EVAL-01/EVAL-02/EVAL-03

Progress: [█████░░░░░] 50% for v1.2 (2/4 phases complete)

## Milestone Metrics

- Planned phases: 4 (13-16)
- Planned requirements: 11
- Plans: 6 complete (Phases 13-14)
- Tasks: 18 complete (plans 13-01..03 and 14-01..03)
- Source context: `deep-research-report.md`

## Accumulated Context

### Decisions Summary

- Streamlit runtime surface is intentionally constrained to five supported operator views.
- Legacy Backtest/Pipeline routes are retirement-guarded with supported-page guidance.
- Home prediction rendering now tolerates missing confidence/optional fields without crashes.
- `make test-release` remains the canonical acceptance command for D1/D2/D3 gates.
- Simplified UI behavior is release-wired by dedicated phase-level regression and smoke suites.
- v1.2 will prioritize leakage control, retraining cadence, and walk-forward rigor over immediate high-complexity model changes.
- Tree-ensemble baselines are the first accuracy benchmark before considering transformer or graph model expansion.
- As-of alignment with no-future-match enforcement is now a dataset-level invariant (DATA-01).
- Return targets (`r_forward`) and direction labels now share one horizon-aware labeling path (DATA-02).
- Triple-barrier event labels are available as an optional dataset/CLI mode while default return-direction behavior is unchanged (LABL-01).
- Phase 14 shipped dual baseline family support (`xgb`, `random_forest`) with comparable persistence artifacts (MODL-01).
- Retraining and CV now use explicit rolling train/backtest windows configurable through CLI/config and autonomous flows (MODL-02).
- Per-window regime/drift diagnostics are emitted in evaluation, retraining artifacts, and monitor CLI output (MODL-03).

### Pending Todos

- Execute Phase 15 (`$gsd-execute-phase 15`) for cost-aware walk-forward evaluation requirements (EVAL-01/02/03).
- Verify Phase 15 outcome and close evaluation rigor requirements.
- Keep v1.2 model promotion tied to out-of-sample and drawdown-safe gates.

### Blockers/Concerns

- No active blockers. Main risk is accidental leakage or overfitting in accuracy optimization loops.

## Session Continuity

Last session: 2026-02-26
Stopped at: Phase 15 planning complete
Resume with: `$gsd-execute-phase 15`
