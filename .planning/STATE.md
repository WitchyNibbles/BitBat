# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-25)

**Core value:** A reliable prediction system where operators can trust that monitoring runs without DB failures and the timeline shows clear prediction vs. outcome history.
**Current focus:** Phase 14 execution prep for baseline model and retraining cadence upgrades

## Current Position

Milestone: v1.2 (BTC Prediction Accuracy Evolution)
Phase: 14-baseline-models-and-retraining-cadence (planned)
Status: Phase 14 plans created and verified; ready for execution
Last activity: 2026-02-26 - Planned Phase 14 from v1.2 requirements and deep-research guidance

Progress: [██░░░░░░░░] 25% for v1.2 (1/4 phases complete)

## Milestone Metrics

- Planned phases: 4 (13-16)
- Planned requirements: 11
- Plans: 6 (Phases 13-14)
- Tasks: 9 complete (plans 13-01, 13-02, and 13-03)
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
- Phase 14 planning uses a tree-baseline-first approach: dual baselines, windowed retraining cadence, and per-window drift diagnostics.

### Pending Todos

- Execute Phase 14 (`$gsd-execute-phase 14`).
- Verify Phase 14 outcome against MODL-01/MODL-02/MODL-03 requirements.
- Keep v1.2 model promotion tied to out-of-sample and drawdown-safe gates.

### Blockers/Concerns

- No active blockers. Main risk is accidental leakage or overfitting in accuracy optimization loops.

## Session Continuity

Last session: 2026-02-25
Stopped at: phase 14 planning complete
Resume with: `$gsd-execute-phase 14`
