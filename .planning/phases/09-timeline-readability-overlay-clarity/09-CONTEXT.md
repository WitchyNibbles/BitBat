# Phase 9: Timeline Readability and Overlay Clarity - Context

**Gathered:** 2026-02-25
**Status:** Ready for planning

<domain>
## Phase Boundary

Restore at-a-glance interpretability of the prediction timeline while preserving
predicted-vs-realized comparison value. This phase is limited to timeline rendering,
timeline controls, and timeline-focused regression gates.

</domain>

<decisions>
## Implementation Decisions

### Readability Priority
- The default timeline view must be understandable in under a few seconds with dense data.
- Visual hierarchy should communicate "what happened" first and detailed analytics second.

### Overlay Behavior
- Predicted-vs-realized comparison remains supported, but must not obscure base timeline context.
- Comparison behavior should be explicit and opt-in from the operator's perspective.

### Scope Boundaries
- Keep existing practical filters (`freq`, `horizon`, `date window`) and summary metrics.
- Focus implementation on `src/bitbat/gui/timeline.py`, `streamlit/pages/0_Quick_Start.py`,
  and GUI timeline tests.

### Claude's Discretion
- Choose whether comparison is presented as a dedicated secondary chart or a mode-based view.
- Tune exact visual styling (line widths, opacities, marker sizes, legend semantics).
- Define specific readability acceptance heuristics in automated tests.

</decisions>

<specifics>
## Specific Ideas

- Current timeline composition appears visually crowded under real data (user screenshot, 2026-02-25).
- Default overlay-on behavior contributes to cognitive load for first-view interpretation.
- Dual-axis + mismatch fill should remain available but not dominate primary event interpretation.

</specifics>

<deferred>
## Deferred Ideas

- Multi-model comparative overlays (ANLY-01).
- Exportable annotated timeline segment reports (ANLY-02).
- Full dashboard redesign outside timeline scope.

</deferred>

---

*Phase: 09-timeline-readability-overlay-clarity*
*Context gathered: 2026-02-25*
