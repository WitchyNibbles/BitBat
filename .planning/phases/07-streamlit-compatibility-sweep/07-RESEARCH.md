# Phase 7: Streamlit Compatibility Sweep - Research

**Researched:** 2026-02-24
**Domain:** Streamlit width API deprecation cleanup and regression-proof compatibility checks
**Confidence:** HIGH

## User Constraints

No `07-CONTEXT.md` was provided. Research is based on roadmap requirements, current codebase state, and existing GUI test patterns.

## Summary

Phase 7 targets Streamlit API hygiene rather than feature behavior changes. The repository already uses modern `width="stretch"` in most GUI surfaces, but active runtime pages still contain deprecated `use_container_width=True` usage.

Current inventory shows 6 deprecated call sites in active UI modules:
- `streamlit/pages/0_Quick_Start.py` (5 call sites)
- `streamlit/pages/4_🔧_System.py` (1 call site)

No `use_container_width=False` usage exists in runtime UI files, so GUI-02 is primarily a compatibility invariant: prevent boolean width arguments and enforce the modern width API contract (`width='stretch'` or `width='content'`).

The lowest-risk implementation path is:
1. Directly migrate known deprecated call sites.
2. Add deterministic source-based compatibility tests covering runtime Streamlit entrypoints (`streamlit/app.py`, `streamlit/pages/*.py`).
3. Add a phase-level regression gate that pairs compatibility checks with primary GUI integration tests to prove warning-free width behavior is preserved.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| GUI-01 | Replace deprecated `use_container_width=True` with `width='stretch'`. | Migrate all active call sites in Quick Start and System pages; enforce zero deprecated usage in runtime files. |
| GUI-02 | Replace deprecated `use_container_width=False` with `width='content'`. | Add invariant checks that reject boolean width API patterns and accept `width='content'` where content-width behavior is needed. |
| GUI-03 | Primary GUI workflows run without width deprecation warnings. | Use compatibility gate + core GUI integration regression suite to keep primary pages warning-free. |

</phase_requirements>

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| streamlit | lockfile currently resolves to 1.53.1 | Width API behavior (`width=...` vs deprecated `use_container_width`) | Primary GUI framework and deprecation source |
| pytest | repo baseline | Automated compatibility and regression checks | Existing test harness used across GUI phases |
| pathlib + ast | stdlib | Deterministic static inspection of Streamlit source files | Fast, dependency-free API usage validation |

### Supporting

| Library | Purpose | When to Use |
|---------|---------|-------------|
| re (stdlib) | Lightweight scan fallback for deprecation tokens | Fast fail checks before AST pass |
| importlib/path discovery helpers in tests | Runtime file inventory for `streamlit/app.py` and pages | Keep checks aligned to current page set |

## Architecture Patterns

### Pattern 1: Runtime-Scope Migration

**What:** Migrate deprecated width arguments only in active runtime GUI surfaces (`streamlit/app.py` and `streamlit/pages/*.py`).
**Why:** Avoid accidental churn in backup/reference files while satisfying user-visible requirements.

### Pattern 2: Static API Contract Gate

**What:** Add source-level tests asserting no `use_container_width` usage and no boolean `width` arguments in runtime files.
**Why:** Guarantees deprecation-free width usage without requiring brittle Streamlit server boot in tests.

### Pattern 3: Phase-Level Compatibility Gate

**What:** Add one phase-level regression test aggregating width compatibility checks and primary GUI integration command.
**Why:** Keeps GUI-03 verification explicit and reusable during later release verification work.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Warning detection | Runtime log scraping from interactive Streamlit sessions | Deterministic source + pytest compatibility checks | CI-friendly, non-flaky verification |
| Broad migration | Blind regex replace across entire repository | Runtime-scope targeted edits + regression guard | Prevents touching non-runtime artifacts unnecessarily |
| Width policy drift | Ad hoc page-by-page review only | Automated scan with explicit allowlist (`stretch`, `content`) | Locks policy and prevents regressions |

## Common Pitfalls

### Pitfall 1: Partial migration

Replacing most but not all `use_container_width` calls leaves hidden deprecation warnings in production pages.

### Pitfall 2: Wrong width literal

Using unsupported values or booleans in `width=` reintroduces compatibility problems even after removing `use_container_width`.

### Pitfall 3: Over-scoping checks

Including non-runtime backup files in hard-fail checks creates false positives and slows phase delivery.

### Pitfall 4: Verification gap

Relying on manual page clicks instead of automated checks allows regressions to reappear between phases.

## Validation Architecture

### Validation Objective

Prove that active GUI workflows no longer rely on deprecated Streamlit width arguments and remain guarded against regressions.

### Validation Layers

1. **Compatibility unit tests (`tests/gui/test_streamlit_width_compat.py`)**
   - no runtime `use_container_width`
   - no boolean `width` arguments
   - allowed width literals only (`stretch`, `content`) where applicable

2. **GUI integration regressions (`tests/gui/test_complete_gui.py`)**
   - primary GUI data and timeline integration remains stable after width API migration

3. **Phase-level gate (`tests/gui/test_phase7_streamlit_compat_complete.py`)**
   - validates combined GUI-01/02/03 semantics with one command

### Automated Gates

- `poetry run pytest tests/gui/test_streamlit_width_compat.py -q`
- `poetry run pytest tests/gui/test_complete_gui.py -q`
- `poetry run pytest tests/gui/test_streamlit_width_compat.py tests/gui/test_complete_gui.py tests/gui/test_phase7_streamlit_compat_complete.py -q`

## Code Evidence (Current State)

```python
# streamlit/pages/0_Quick_Start.py
if st.button("Train Model", type="primary", use_container_width=True):
...
st.plotly_chart(fig, use_container_width=True)
...
if st.button("Retrain Model", type="primary", use_container_width=True):
```

```python
# streamlit/pages/4_🔧_System.py
if st.button("Save Autonomous Settings", use_container_width=True):
```

All other active Streamlit pages already predominantly use `width="stretch"` where full-width rendering is intended.

## Recommended Plan Split

1. **07-01 (Wave 1):** Migrate deprecated width arguments in runtime pages and add compatibility regression tests.
2. **07-02 (Wave 2):** Add phase-level warning-free compatibility gate for primary GUI workflows.

## Sources

### Primary (HIGH confidence)
- `streamlit/pages/0_Quick_Start.py`
- `streamlit/pages/4_🔧_System.py`
- `streamlit/app.py`
- `tests/gui/test_complete_gui.py`
- `.planning/ROADMAP.md`
- `.planning/REQUIREMENTS.md`
- `.planning/STATE.md`

### Secondary (MEDIUM confidence)
- `pyproject.toml`
- `poetry.lock`

## Metadata

**Confidence breakdown:**
- Width API migration scope: HIGH
- Compatibility gate design: HIGH
- Regression strategy for GUI-03: HIGH

**Research date:** 2026-02-24
**Valid until:** 2026-03-24
