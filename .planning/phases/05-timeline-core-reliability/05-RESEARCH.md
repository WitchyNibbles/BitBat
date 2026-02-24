# Phase 5: Timeline Core Reliability - Research

**Researched:** 2026-02-24
**Domain:** Streamlit timeline reliability for mixed pending/realized prediction records
**Confidence:** HIGH

## User Constraints

No `05-CONTEXT.md` was provided. Research is constrained by roadmap goals, requirements (`TIM-01`, `TIM-02`), and existing phase artifacts.

## Summary

Phase 5 should harden timeline read and render behavior so it remains trustworthy across real monitor data, including mixed pending/realized rows and nullable prediction fields. Current timeline behavior is still coupled to legacy probability semantics (`p_up`/`p_down`) even though monitor/API contracts now center on `predicted_return` and `predicted_price`.

The current `get_timeline_data` and `build_timeline_figure` path has three reliability gaps: (1) confidence and hover semantics still derive from probability fields that can be null or fabricated defaults, (2) marker rendering can silently disappear when price points are missing or sparse, and (3) pending vs realized outcome state is encoded indirectly (opacity only) instead of a normalized status contract consumed consistently by the Streamlit page.

**Primary recommendation:** add a normalized timeline read-model in `src/bitbat/gui/timeline.py`, make the figure builder status-driven and resilient to partial price coverage, then lock behavior with mixed-fixture regression tests that represent real operational data shapes.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| TIM-01 | Timeline page renders recent and historical prediction records reliably from operational data. | Add schema-tolerant read-model normalization + robust rendering fallback for sparse/nullable data. |
| TIM-02 | Timeline clearly distinguishes pending vs realized predictions and shows realized outcome alignment. | Derive explicit prediction status (`pending`, `realized_correct`, `realized_wrong`) and render status-aware marker/hover behavior. |
</phase_requirements>

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pandas | repo baseline | Timeline read-model normalization and timestamp shaping | Existing timeline and tests already depend on DataFrame-first transforms |
| sqlite3 | stdlib | Local prediction-history reads from `autonomous.db` | Local-first runtime architecture and current timeline code path |
| plotly.graph_objects | repo baseline | Interactive timeline rendering in Streamlit | Existing timeline figure already built with Plotly |
| Streamlit | repo baseline | Timeline page rendering and summary metrics | Existing `0_Quick_Start.py` timeline fragment |
| pytest | repo baseline | Regression safety for mixed fixtures and rendering behavior | Existing GUI test suite already in place |

### Supporting

| Library | Purpose | When to Use |
|---------|---------|-------------|
| numpy | fixture generation and deterministic ranges | Test fixture matrix for large mixed datasets |
| pathlib | deterministic path handling for local artifacts | File-path logic in timeline readers/tests |

## Architecture Patterns

### Pattern 1: Normalized Timeline Read-Model (Single Contract)

**What:** Convert DB rows into one canonical DataFrame contract before any rendering.
**When to use:** Immediately after SQL read in `get_timeline_data`.
**Required normalized fields:**
- `timestamp_utc` (parsed, timezone-safe)
- `predicted_direction`
- `predicted_return` (nullable)
- `predicted_price` (nullable)
- `confidence` (nullable, derived from probabilities only when meaningful)
- `prediction_status` (`pending`, `realized_correct`, `realized_wrong`)
- `is_realized` (bool)

### Pattern 2: Status-First Rendering

**What:** Figure styling is keyed by normalized status, not ad hoc null checks.
**When to use:** Marker color/opacity/hover generation in `build_timeline_figure`.
**Outcome:** Pending vs realized states are visually explicit and deterministic.

### Pattern 3: Price-Series Fallback for Marker Placement

**What:** If nearest market price is unavailable, use safe fallback (`predicted_price` when present) rather than dropping event markers.
**When to use:** Marker y-position resolution inside `build_timeline_figure`.
**Outcome:** Timeline remains populated even when raw price coverage is partial.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Per-page SQL shape logic | Duplicate query/normalization in Streamlit pages | Centralize in `src/bitbat/gui/timeline.py` | Prevents timeline drift across views |
| Pending/realized inference in UI layer | Repeated null/boolean checks in page code | Normalized `prediction_status` in read-model | Keeps semantics testable and consistent |
| Confidence synthesis from arbitrary defaults | Hardcoded `max(0, 0)` fallback confidence | Nullable confidence with explicit display behavior | Avoids misleading “0% confidence” output |

## Common Pitfalls

### Pitfall 1: Dropping markers when prices are sparse
`build_timeline_figure` currently skips markers when no nearest price exists. This causes empty/partial charts despite available prediction events.

### Pitfall 2: Mixed schema semantics in one timeline
Legacy rows may carry probabilities without return/price, while newer rows carry return/price semantics. Without normalization, hover text and metrics become inconsistent.

### Pitfall 3: Boolean correctness interpretation drift
`correct` may arrive as `None`, `0/1`, or bool depending on query path. Status derivation must normalize this consistently.

## Validation Architecture

### Validation Objective
Guarantee timeline rendering reliability for realistic operational datasets, not only idealized fixtures.

### Validation Layers

1. **Read-model normalization tests (`tests/gui/test_timeline.py`)**
   - Mixed pending/realized rows
   - Mixed legacy/new semantic fields
   - Freq/horizon filtering and ordering guarantees

2. **Figure behavior tests (`tests/gui/test_timeline.py`)**
   - Status-aware marker styling
   - Missing/partial price fallback behavior
   - Non-empty output when predictions exist but price coverage is incomplete

3. **GUI integration tests (`tests/gui/test_complete_gui.py` + phase test)**
   - Summary metric consistency (total/completed/correct)
   - No crash behavior for representative timeline fixture matrices

### Automated Gates
- `poetry run pytest tests/gui/test_timeline.py -q`
- `poetry run pytest tests/gui/test_complete_gui.py -q -k "prediction or status or timeline"`
- `poetry run pytest tests/gui/test_timeline.py tests/gui/test_complete_gui.py -q`

## Code Evidence (Current State)

```python
# src/bitbat/gui/timeline.py (current)
SELECT timestamp_utc, predicted_direction, p_up, p_down,
       actual_return, actual_direction, correct
...
confidence = max(float(row.get("p_up", 0)), float(row.get("p_down", 0)))
```

```python
# streamlit/pages/0_Quick_Start.py (current)
predictions = get_timeline_data(_DB_PATH, freq, horizon)
fig = build_timeline_figure(predictions, prices)
realized = predictions["correct"].notna().sum()
```

Both paths depend on unnormalized timeline columns and should move to an explicit status/read-model contract.

## Recommended Plan Split

1. **05-01 (Wave 1):** Build normalized timeline read-model layer with schema-tolerant prediction semantics.
2. **05-02 (Wave 2):** Repair timeline rendering for mixed pending/realized + sparse price coverage and align Streamlit consumer.
3. **05-03 (Wave 3):** Add representative fixture regressions and phase-level validation gate coverage.

## Sources

### Primary (HIGH confidence)
- `src/bitbat/gui/timeline.py`
- `streamlit/pages/0_Quick_Start.py`
- `tests/gui/test_timeline.py`
- `tests/gui/test_complete_gui.py`
- `.planning/phases/04-monitor-flow-consistency-api-alignment/04-02-SUMMARY.md`
- `.planning/phases/04-monitor-flow-consistency-api-alignment/04-VERIFICATION.md`
- `.planning/ROADMAP.md`
- `.planning/REQUIREMENTS.md`

### Secondary (MEDIUM confidence)
- `src/bitbat/gui/widgets.py` (semantic alignment context)
- `src/bitbat/autonomous/models.py` (prediction outcome field contract)

## Metadata

**Confidence breakdown:**
- Timeline read-model drift diagnosis: HIGH
- Pending/realized rendering gap diagnosis: HIGH
- Mixed-fixture regression strategy: HIGH

**Research date:** 2026-02-24
**Valid until:** 2026-03-24
