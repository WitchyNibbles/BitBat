# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-25)

**Core value:** A reliable prediction system where operators can trust that monitoring runs without DB failures and the timeline shows clear prediction vs. outcome history.
**Current focus:** Milestone v1.2 prediction-accuracy planning and phase 13 kickoff

## Current Position

Milestone: v1.2 (BTC Prediction Accuracy Evolution)
Phase: 13-data-and-label-contract-upgrade (not started)
Status: v1.2 initialized; requirements and roadmap defined
Last activity: 2026-02-25 - Started v1.2 using deep-research-driven accuracy strategy

Progress: [░░░░░░░░░░] 0% for v1.2 (0/4 phases complete)

## Milestone Metrics

- Planned phases: 4 (13-16)
- Planned requirements: 11
- Plans: 0 (phase planning not started)
- Tasks: 0 (execution not started)
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

### Pending Todos

- Create execution plan for Phase 13 (`$gsd-plan-phase 13`).
- Implement v1.2 Phase 13 data/label contract upgrades after planning approval.
- Keep v1.2 model promotion tied to out-of-sample and drawdown-safe gates.

### Blockers/Concerns

- No active blockers. Main risk is accidental leakage or overfitting in accuracy optimization loops.

## Session Continuity

Last session: 2026-02-25
Stopped at: v1.2 milestone initialization and roadmap setup
Resume with: `$gsd-plan-phase 13`
