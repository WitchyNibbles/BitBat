# Phase 6: Timeline UX Expansion (T2) - Research

**Researched:** 2026-02-24
**Domain:** Timeline UX expansion for confidence/direction clarity, filtering, and predicted-vs-realized overlays
**Confidence:** HIGH

## User Constraints (from 06-CONTEXT.md)

- Confidence remains hover-only, shown as exact percentage, with explicit `n/a` when unavailable.
- Direction remains encoded with both color and shape.
- Filters (`freq`, `horizon`, date window) are always visible and persist in session state.
- Date window defaults to last 7 days; no-result combinations must show explicit empty-state messaging without implicit auto-reset.
- Overlay comparison uses predicted + realized lines with a subtle mismatch band, pending rows rendered as predicted-only, and legend toggles for component visibility.
- Base chart density remains minimal; rich detail belongs in hover and a compact summary strip.

## Summary

Phase 5 established a stable timeline data contract (`prediction_status`, normalized `correct`, nullable `confidence`) and robust marker placement fallback behavior. Phase 6 should build directly on that contract by adding UX controls and comparison surfaces, not by altering foundational normalization semantics.

The key design constraint is avoiding semantic drift between timeline rendering and summary behavior. The existing `build_timeline_figure` function currently combines price line + event markers in one chart; Phase 6 should add clearly scoped helper layers:
- filter/window shaping before figure construction,
- status-aware summary metrics bound to active filters,
- overlay dataset construction for predicted-vs-realized returns with pending-safe gaps.

Because requirements TIM-03/04/05 are user-visible and interactive, the best risk-reduction strategy is to centralize filter and overlay transformation logic in `src/bitbat/gui/timeline.py`, keep Streamlit page controls thin, and lock behavior with deterministic fixture tests.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| TIM-03 | Timeline exposes confidence and direction context per prediction event. | Keep marker view minimal while strengthening hover payload and compact status/confidence summary strip. |
| TIM-04 | Timeline supports practical filtering (freq/horizon/date window) without breaking rendering. | Add explicit filter application helpers and empty-state messaging semantics with persistent session-state controls. |
| TIM-05 | Timeline supports improved predicted-vs-realized visualization. | Add overlay data construction and plot traces (predicted/realized lines + mismatch band) with pending-safe behavior. |

</phase_requirements>

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pandas | repo baseline | Timeline shaping, filter windows, overlay datasets | Existing timeline data path is DataFrame-native |
| plotly.graph_objects | repo baseline | Timeline + overlay traces and legend toggles | Existing figure construction already in Plotly |
| streamlit | repo baseline | Filter controls, session persistence, compact summaries | Existing UI integration point (`0_Quick_Start.py`) |
| sqlite3 | stdlib | Operational prediction-history reads | Existing local-first architecture |
| pytest | repo baseline | Regression coverage for filters + overlays + UI integration | Existing GUI regression suite |

### Supporting

| Library | Purpose | When to Use |
|---------|---------|-------------|
| datetime/pandas Timedelta | Date window mapping (`24h`, `7d`, `30d`) | Filter window transformation and boundary testing |
| numpy | Stable fixture generation for overlay mismatch scenarios | Deterministic synthetic test matrix setup |

## Architecture Patterns

### Pattern 1: Filter-First Read Model Consumption

**What:** Apply freq/horizon/date-window filtering to normalized timeline rows before chart assembly and summary metrics.
**Where:** `src/bitbat/gui/timeline.py` helper(s), consumed by `streamlit/pages/0_Quick_Start.py`.
**Why:** Prevent chart and metric divergence under active filters.

### Pattern 2: Minimal Surface, Rich Hover

**What:** Keep chart markers low-noise, put confidence/return detail in hover payload and compact summary strip.
**Where:** `build_timeline_figure` hovertemplate + Quick Start summary block.
**Why:** Matches locked context decisions and avoids over-annotated timelines.

### Pattern 3: Overlay Dataset Contract

**What:** Construct a dedicated overlay DataFrame keyed by timestamp with:
- `predicted_return`,
- `actual_return`,
- `prediction_status`,
- optional mismatch magnitude.
Pending rows must preserve predicted values and leave realized values null.
**Where:** Timeline module helper used by overlay trace builder.
**Why:** Encodes TIM-05 behavior once, testable independent of Streamlit controls.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Per-widget filter math | Inline date/freq/horizon slicing in page code | Shared filter helper in timeline module | Prevents UI logic drift and duplicate bugs |
| Overlay logic tied to UI widgets | Ad hoc trace mutation in Streamlit callback | Pure overlay dataset builder + deterministic trace config | Keeps tests simple and robust |
| No-result filter handling via silent resets | Auto-changing user-selected filters | Explicit empty-state message under user-selected filters | Preserves user intent and debuggability |

## Common Pitfalls

### Pitfall 1: Overlay scale confusion
Price and return units differ significantly. Overlay should compare predicted vs realized returns (same unit), not blend returns into BTC price axis.

### Pitfall 2: Filter and summary mismatch
If summary metrics are computed from unfiltered rows while chart uses filtered rows, users see contradictory analytics.

### Pitfall 3: Pending row semantic leakage
Filling realized values for pending rows (or drawing zero-value realized traces) misrepresents outcomes and breaks TIM-05 correctness.

### Pitfall 4: Session-state resets on rerun
Uninitialized/default-overwrite patterns in Streamlit can unintentionally reset filter state every rerun unless keys are guarded.

## Validation Architecture

### Validation Objective
Guarantee that UX expansion features remain semantically correct and resilient under operationally realistic filter + overlay scenarios.

### Validation Layers

1. **Timeline unit tests (`tests/gui/test_timeline.py`)**
   - filter window boundaries,
   - overlay dataset construction semantics,
   - pending/realized overlay behavior,
   - confidence hover value handling (`%` or `n/a`).

2. **GUI integration tests (`tests/gui/test_complete_gui.py`)**
   - filtered summary metrics alignment,
   - session-state persistence expectations,
   - no-result filter-state messaging contract.

3. **Phase-level gate (`tests/gui/test_phase6_timeline_ux_complete.py`)**
   - end-to-end filtered + overlay timeline behavior,
   - legend toggle trace presence,
   - mismatch band and pending-safe rendering.

### Automated Gates

- `poetry run pytest tests/gui/test_timeline.py -q -k "timeline and (filter or overlay or confidence)"`
- `poetry run pytest tests/gui/test_complete_gui.py -q -k "timeline and (filter or status or summary)"`
- `poetry run pytest tests/gui/test_timeline.py tests/gui/test_complete_gui.py tests/gui/test_phase6_timeline_ux_complete.py -q`

## Code Evidence (Current State)

```python
# src/bitbat/gui/timeline.py (current)
def get_timeline_data(db_path, freq, horizon, limit=168) -> pd.DataFrame:
    ...
    return _normalize_timeline_rows(raw_df)

def build_timeline_figure(predictions, prices) -> object:
    # currently price line + status markers, no filter controls/overlay traces
```

```python
# streamlit/pages/0_Quick_Start.py (current)
predictions = get_timeline_data(_DB_PATH, freq, horizon)
fig = build_timeline_figure(predictions, prices)
status_summary = summarize_timeline_status(predictions)
```

Current flow already provides a stable semantic baseline but lacks TIM-04 controls and TIM-05 overlay comparisons.

## Recommended Plan Split

1. **06-01 (Wave 1):** Implement confidence/direction presentation refinements + compact summary strip semantics and regressions.
2. **06-02 (Wave 2):** Implement always-visible filter controls with session persistence and explicit no-result behavior.
3. **06-03 (Wave 3):** Implement predicted-vs-realized overlay construction/rendering and lock phase-level regression gate.

## Sources

### Primary (HIGH confidence)
- `src/bitbat/gui/timeline.py`
- `streamlit/pages/0_Quick_Start.py`
- `tests/gui/test_timeline.py`
- `tests/gui/test_complete_gui.py`
- `tests/gui/test_phase5_timeline_complete.py`
- `.planning/phases/06-timeline-ux-expansion-t2/06-CONTEXT.md`
- `.planning/ROADMAP.md`
- `.planning/REQUIREMENTS.md`

### Secondary (MEDIUM confidence)
- `.planning/phases/05-timeline-core-reliability/05-RESEARCH.md`
- `.planning/phases/05-timeline-core-reliability/05-VERIFICATION.md`

## Metadata

**Confidence breakdown:**
- Phase boundary and decision alignment: HIGH
- Filter/overlay architecture strategy: HIGH
- Test gate recommendations: HIGH

**Research date:** 2026-02-24
**Valid until:** 2026-03-24
