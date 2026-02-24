# Phase 6: Timeline UX Expansion (T2) - Context

**Gathered:** 2026-02-24
**Status:** Ready for planning

<domain>
## Phase Boundary

Add richer timeline analysis context without regressing reliability by improving confidence/direction presentation, adding practical timeline filters (freq/horizon/date window), and introducing predicted-vs-realized comparison overlays.

</domain>

<decisions>
## Implementation Decisions

### Signal Presentation
- Confidence is shown in hover only (not persistently on marker text).
- Direction is encoded with both color and shape.
- Hover confidence uses exact numeric percentage values.
- If confidence is unavailable, show `n/a` in hover and do not add marker-level confidence cues.

### Filter Behavior
- Freq, horizon, and date window filters are always visible.
- Default date window is last 7 days.
- If no events match current filters, keep chart frame visible and show a clear empty-state message for current filters.
- Filter selections persist in session state during the user session.

### Overlay Comparison
- Primary comparison visualization is two overlaid lines (predicted and realized).
- Show subtle fill/band between lines to indicate mismatch magnitude.
- For pending predictions, render predicted series only; omit realized segment/point and mark pending in hover.
- Overlay components are user-toggleable via simple legend toggles.

### Detail Density
- Base chart view remains minimal: direction + status only.
- Hover must include timestamp, direction, confidence, predicted return, actual return, and status.
- Include a compact summary strip for current filtered context (for example counts, accuracy, average confidence).
- On smaller screens, hide non-critical summary metrics first while preserving chart + hover fidelity.

### Claude's Discretion
- Exact visual styling values (spacing, font sizing, color tokens) within the above semantic rules.
- Exact wording of helper/empty-state copy as long as meaning remains clear.
- Precise responsive breakpoints for when summary metrics collapse.

</decisions>

<specifics>
## Specific Ideas

- Keep default chart readability high by putting dense analytics in hover and summary strip rather than marker text.
- Preserve semantic correctness for pending predictions: no implied realized values.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 06-timeline-ux-expansion-t2*
*Context gathered: 2026-02-24*
