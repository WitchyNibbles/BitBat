# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-25)

**Core value:** A reliable prediction system where operators can trust that monitoring runs without DB failures and the timeline shows clear prediction vs. outcome history.
**Current focus:** Phase 13 execution (plans 13-01 and 13-02 complete; 13-03 next)

## Current Position

Milestone: v1.2 (BTC Prediction Accuracy Evolution)
Phase: 13-data-and-label-contract-upgrade (executing)
Status: Phase 13 in progress with 2/3 plans complete (DATA-01 and DATA-02 delivered)
Last activity: 2026-02-25 - Executed plan 13-02 and updated shared return/direction label contract

Progress: [░░░░░░░░░░] 0% for v1.2 (0/4 phases complete)

## Milestone Metrics

- Planned phases: 4 (13-16)
- Planned requirements: 11
- Plans: 3 (Phase 13)
- Tasks: 6 complete (plans 13-01 and 13-02)
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

### Pending Todos

- Execute remaining Phase 13 plan (`13-03`) for optional triple-barrier label mode (LABL-01).
- Verify completed Phase 13 outcome against DATA-01/DATA-02/LABL-01 requirements.
- Keep v1.2 model promotion tied to out-of-sample and drawdown-safe gates.

### Blockers/Concerns

- No active blockers. Main risk is accidental leakage or overfitting in accuracy optimization loops.

## Session Continuity

Last session: 2026-02-25
Stopped at: phase 13 plan 13-02 complete
Resume with: `$gsd-execute-phase 13`
