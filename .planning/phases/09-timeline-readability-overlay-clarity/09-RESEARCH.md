# Phase 9: Timeline Readability and Overlay Clarity - Research

**Researched:** 2026-02-25
**Domain:** Timeline readability and comparison UX for dense prediction history
**Confidence:** HIGH

<user_constraints>
## User Constraints (from 09-CONTEXT.md)

### Locked Decisions
- Default timeline must be quickly understandable under dense operational data.
- Overlay comparison must remain available but should not obscure base timeline interpretation.
- Existing practical filters and summary metrics remain in scope and must continue to work.
- Scope is limited to timeline rendering surfaces and timeline regression gates.

### Claude's Discretion
- Whether comparison is implemented as a dedicated secondary chart or a mode switch.
- Exact visual hierarchy implementation details (trace grouping, color/opacity, legend behavior).
- Concrete readability acceptance criteria used in tests.

### Deferred Ideas (OUT OF SCOPE)
- Multi-model comparative overlays and export/report capabilities.
- Full dashboard redesign.

</user_constraints>

<research_summary>
## Summary

The current timeline implementation is semantically correct but visually overloaded in dense datasets.
Two design choices contribute most to interpretability loss: one marker trace per prediction row and
default overlay-on behavior in Quick Start. This creates high trace count, competing visual channels,
and dual-axis context switching in the default view.

For Phase 9, the standard approach is to separate "event readability" from "return comparison":
keep the primary chart focused on BTC price and status/direction events, and make return comparison
explicitly opt-in. In code terms, this means (1) consolidating marker traces by semantic groups rather
than per-row traces, and (2) rendering comparison in a dedicated visualization path instead of always
competing with the primary chart.

**Primary recommendation:** Deliver a readability-first default timeline, then expose predicted-vs-realized
comparison through an explicit secondary view with targeted regression gates for interpretability.
</research_summary>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| TIM-03 | Timeline exposes confidence and direction context per prediction event. | Preserve detailed hover semantics while reducing chart clutter and improving legend/readability behavior. |
| TIM-05 | Timeline supports improved visualization for predicted vs realized behavior. | Keep comparison view available but decouple it from the default primary event interpretation surface. |

</phase_requirements>

<standard_stack>
## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| plotly.graph_objects | repo baseline | Timeline and comparison figure rendering | Existing chart stack and test fixtures are plotly-native |
| pandas | repo baseline | Timeline normalization and comparison-frame shaping | Existing timeline data contract is DataFrame-based |
| streamlit | repo baseline | Timeline controls and chart composition | Quick Start already hosts operational timeline workflow |
| pytest | repo baseline | Regression protection for timeline behavior | Existing GUI timeline suites already cover semantic baseline |

### Supporting
| Library | Purpose | When to Use |
|---------|---------|-------------|
| plotly.subplots | Multi-panel chart composition | If comparison is kept in same figure but separate panel |
| numpy/pandas fixture tooling | Deterministic dense-history fixtures | Readability regression scenarios with many timeline events |

</standard_stack>

<architecture_patterns>
## Architecture Patterns

### Pattern 1: Semantic Trace Consolidation

**What:** Build marker traces by grouped semantics (direction + status), not one trace per event row.
**Where:** `src/bitbat/gui/timeline.py` in timeline figure construction.
**Why:** Reduces trace count and legend/hover complexity under dense data.

### Pattern 2: Readability-First Primary View

**What:** Keep primary timeline dedicated to price and prediction-event interpretation.
**Where:** `build_timeline_figure(...)` and Quick Start timeline section.
**Why:** Operators need immediate "what happened" comprehension before deeper diagnostics.

### Pattern 3: Explicit Comparison Mode

**What:** Render predicted-vs-realized comparison via explicit operator action (toggle/mode), ideally as a
dedicated comparison chart path.
**Where:** `streamlit/pages/0_Quick_Start.py` + timeline comparison helper(s).
**Why:** Preserves TIM-05 without overwhelming default TIM-03 readability.

### Pattern 4: Readability Acceptance Tests

**What:** Add tests for interpretability proxies (trace count bounds, default overlay state, label clarity,
and pending-safe comparison semantics).
**Where:** `tests/gui/test_timeline.py`, `tests/gui/test_complete_gui.py`, `tests/gui/test_phase9_*.py`.
**Why:** Existing tests verify semantic correctness but not readability outcomes under dense datasets.

</architecture_patterns>

<dont_hand_roll>
## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Dense marker manageability | Per-point trace mutation loops for every marker scenario | Semantic grouping + shared hover template/customdata | Lower trace count and predictable legend behavior |
| Comparison readability | Always-on dual-axis overlay mixed into primary event chart | Dedicated comparison mode/chart | Avoids axis-context switching in default interpretation path |
| Readability validation | Manual visual QA only | Deterministic fixture tests with measurable assertions | Prevents repeated regressions after future timeline changes |

</dont_hand_roll>

<common_pitfalls>
## Common Pitfalls

### Pitfall 1: Dual-axis overload in default view
**What goes wrong:** Price trend and return comparison compete visually in one chart.
**How to avoid:** Keep comparison opt-in and isolated from default event view.

### Pitfall 2: Marker trace explosion
**What goes wrong:** One trace per event causes visual and performance degradation with longer windows.
**How to avoid:** Group markers into semantic buckets and reuse traces.

### Pitfall 3: Semantic drift between timeline and tests
**What goes wrong:** Structural tests pass while users still cannot interpret chart quickly.
**How to avoid:** Add readability acceptance checks tied to dense-history fixtures.

</common_pitfalls>

## Validation Architecture

### Validation Objective
Ensure timeline remains both semantically correct and human-readable in dense operational conditions.

### Validation Layers
1. **Unit (`tests/gui/test_timeline.py`)**
   - Marker grouping/readability semantics.
   - Comparison trace semantics (including pending rows).
2. **Integration (`tests/gui/test_complete_gui.py`)**
   - Default control state and visibility behavior in Quick Start timeline.
   - Summary/readability behavior under active filters.
3. **Phase gate (`tests/gui/test_phase9_timeline_readability_complete.py`)**
   - End-to-end dense fixture assertions for readability + comparison interaction.

### Automated Gates
- `poetry run pytest tests/gui/test_timeline.py -q -k "timeline and (readability or overlay or confidence or marker)"`
- `poetry run pytest tests/gui/test_complete_gui.py -q -k "timeline and (overlay or default or summary or filter)"`
- `poetry run pytest tests/gui/test_timeline.py tests/gui/test_complete_gui.py tests/gui/test_phase9_timeline_readability_complete.py -q`

## Code Evidence (Current State)

```python
# src/bitbat/gui/timeline.py
# - Adds one marker trace per row (dense trace count at scale)
# - Overlay traces share the same figure with dual-axis layout
```

```python
# streamlit/pages/0_Quick_Start.py
# - timeline_show_overlay defaults to True
# - default rendering calls build_timeline_figure(..., show_overlay=True/False toggle)
```

## Recommended Plan Split

1. **09-01 (Wave 1):** Refactor primary timeline rendering for readability (semantic marker grouping + visual hierarchy).
2. **09-02 (Wave 2):** Rework comparison mode behavior and defaults (overlay opt-in, clearer comparison presentation).
3. **09-03 (Wave 3):** Add readability-focused regression gates and integrate phase-level coverage.

## Sources

### Primary (HIGH confidence)
- `src/bitbat/gui/timeline.py`
- `streamlit/pages/0_Quick_Start.py`
- `tests/gui/test_timeline.py`
- `tests/gui/test_complete_gui.py`
- `tests/gui/test_phase6_timeline_ux_complete.py`
- `.planning/v1.0-MILESTONE-AUDIT.md`
- `.planning/ROADMAP.md`
- `.planning/REQUIREMENTS.md`
- `.planning/phases/09-timeline-readability-overlay-clarity/09-CONTEXT.md`

## Metadata

**Confidence breakdown:**
- Readability risk diagnosis: HIGH
- Recommended architecture split: HIGH
- Regression strategy: HIGH

**Research date:** 2026-02-25
**Valid until:** 2026-03-27

---

*Phase: 09-timeline-readability-overlay-clarity*
*Research completed: 2026-02-25*
*Ready for planning: yes*
