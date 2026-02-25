# Phase 10: Supported Surface Pruning - Context

**Gathered:** 2026-02-25
**Status:** Ready for planning

<domain>
## Phase Boundary

Reduce the active Streamlit experience to the five operator-used views
(Quick Start, Settings, Performance, About, System) and remove non-core pages
from normal navigation. This phase is focused on UI surface simplification,
copy/link alignment, and baseline contract checks for the supported view set.

</domain>

<decisions>
## Implementation Decisions

### Keep Only Actively Used Views
- Supported runtime views for this milestone are: `0_Quick_Start`, `1_⚙️_Settings`,
  `2_📈_Performance`, `3_ℹ️_About`, and `4_🔧_System`.
- Non-core pages (`5_🔔_Alerts`, `6_📊_Analytics`, `7_📅_History`, `8_🎯_Backtest`,
  `9_🔬_Pipeline`) are not part of the normal runtime surface.

### Navigation and Link Contract
- Home page quick actions must route only to supported views.
- About/help copy must stop directing users to retired advanced pages.
- Supported-surface decisions should be explicit and test-locked.

### Scope Boundaries
- Phase 10 does not re-implement broken advanced workflows.
- Runtime crash hardening for legacy routes and broken advanced imports is handled in Phase 11.

### Claude's Discretion
- Choose the safest pruning strategy (file relocation, retirement stubs, or navigation contract layer)
  as long as normal runtime surface only shows supported views.
- Define pragmatic contract assertions in tests to lock supported-page inventory.

</decisions>

<specifics>
## Specific Ideas

- User only finds practical value in five views: Quick Start, Settings, Performance, About, System.
- Reported broken pages to retire from default surface include:
  - Home app crash with `KeyError: 'confidence'`
  - Backtest runtime indexing failure
  - Pipeline import failure for `classification_metrics`
- The simplification milestone should favor clarity and reliability over preserving legacy page breadth.

</specifics>

<deferred>
## Deferred Ideas

- Reintroduction of advanced interfaces after stable API/UX contracts are defined.
- New advanced analytics/backtest/pipeline capabilities (covered by deferred ANLY requirements).

</deferred>

---

*Phase: 10-supported-surface-pruning*
*Context gathered: 2026-02-25*
