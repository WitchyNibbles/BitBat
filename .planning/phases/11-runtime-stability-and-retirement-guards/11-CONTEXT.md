# Phase 11: Runtime Stability and Retirement Guards - Context

**Gathered:** 2026-02-25
**Status:** Ready for planning

<domain>
## Phase Boundary

Harden the simplified five-view runtime so startup and home rendering do not crash
when prediction data is partial, and ensure legacy advanced-page entrypoints fail
gracefully with user-facing retirement guidance instead of tracebacks.

</domain>

<decisions>
## Implementation Decisions

### Protect the Active Runtime First
- Keep the active UI surface limited to Quick Start, Settings, Performance, About, and System.
- Prioritize startup and home stability over restoring retired advanced workflows.

### Guard Partial Prediction Rows
- Home rendering must tolerate missing prediction fields, especially `confidence`.
- Missing optional fields should render safe fallback UI text instead of raising exceptions.

### Retire Legacy Routes Safely
- Legacy Backtest and Pipeline entrypoints should display a retirement notice or redirect guidance.
- Retired route access must not import broken advanced-only modules in normal usage paths.

### Scope Boundaries
- Phase 11 hardens runtime behavior and retirement UX.
- Rebuilding advanced Backtest/Pipeline product features remains deferred.

### Claude's Discretion
- Choose pragmatic fallback defaults for confidence display as long as no traceback occurs.
- Choose notice-vs-redirect UX details as long as retired access is user-friendly and test-locked.

</decisions>

<specifics>
## Specific Ideas

- Reported failure signatures to eliminate in this phase:
  - `KeyError: 'confidence'` in `streamlit/app.py` home rendering.
  - Pipeline import crash: `cannot import name 'classification_metrics'`.
  - Backtest runtime indexing crash: `too many indices for array`.
- Standard navigation already excludes retired pages; this phase adds explicit guard behavior for legacy entry paths.

</specifics>

<deferred>
## Deferred Ideas

- Reintroducing advanced Analytics/History/Backtest/Pipeline workflows.
- Full redesign of retirement UX beyond concise guidance and safe redirect behavior.

</deferred>

---

*Phase: 11-runtime-stability-and-retirement-guards*
*Context gathered: 2026-02-25*
