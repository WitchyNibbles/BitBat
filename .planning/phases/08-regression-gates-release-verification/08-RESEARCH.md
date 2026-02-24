# Phase 8: Regression Gates & Release Verification - Research

**Researched:** 2026-02-24
**Domain:** Release-grade regression gates for D1 (schema+monitor), D2 (timeline behavior), D3 (Streamlit width compatibility)
**Confidence:** HIGH

## User Constraints

No `08-CONTEXT.md` exists. This research is grounded in roadmap/requirements plus current code and test coverage.

## Summary

Phases 1-7 established strong domain coverage, but release criteria are still distributed across many test modules without a single Phase 8 acceptance structure. Existing suites already validate most primitives:

- **D1 foundations exist** across monitor/schema/CLI/API tests (`tests/autonomous/test_agent_integration.py`, `tests/test_cli.py`, `tests/api/test_health.py`, `tests/api/test_metrics.py`).
- **D2 foundations exist** across timeline unit + phase tests (`tests/gui/test_timeline.py`, `tests/gui/test_phase5_timeline_complete.py`, `tests/gui/test_phase6_timeline_ux_complete.py`, `tests/gui/test_complete_gui.py`).
- **D3 foundations exist** from Phase 7 (`tests/gui/test_streamlit_width_compat.py`, `tests/gui/test_phase7_streamlit_compat_complete.py`).

The Phase 8 objective is not inventing new behavior; it is consolidating these guarantees into explicit regression gates and a final acceptance pass so D1/D2/D3 remain enforced as a release contract.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| QUAL-01 | Automated tests cover DB schema compatibility + monitor stability regressions for D1. | Add a dedicated D1 phase-level gate test and align CLI/API/agent assertions into one canonical D1 command. |
| QUAL-02 | Automated tests cover timeline data/render behavior for D2. | Add a D2 phase-level timeline gate that composes Phase 5 + 6 invariants and lock one canonical D2 command. |
| QUAL-03 | Automated checks prevent reintroduction of `use_container_width` usage for D3. | Keep/expand AST-based width guards, wire final release acceptance target that includes D3 checks plus D1/D2 suites. |

</phase_requirements>

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pytest | repo baseline | Regression and release-gate execution | Existing test framework used by all current phases |
| sqlite3 + SQLAlchemy | stdlib + repo baseline | Schema/monitor compatibility fixture simulation | Existing monitor/storage contract already uses these layers |
| pandas + plotly | repo baseline | Timeline behavior and render semantics tests | Existing timeline coverage built on DataFrame + Plotly traces |
| ast + pathlib | stdlib | Static Streamlit API guard checks | Deterministic D3 compatibility enforcement without UI runtime flakiness |

### Supporting

| Library | Purpose | When to Use |
|---------|---------|-------------|
| make | Repeatable local acceptance command (`test-release`) | Final Phase 8 release verification workflow |
| httpx test clients (`SyncASGIClient`) | API endpoint regression checks | D1 schema-readiness and metrics behavior assertions |

## Architecture Patterns

### Pattern 1: Phase-Level Gate Modules per Release Dimension

**What:** Add dedicated `test_phase8_*_complete.py` modules for D1 and D2 (and final release-gate orchestration for D3-inclusive acceptance).
**Why:** Makes release criteria explicit and discoverable, rather than inferred from scattered tests.

### Pattern 2: Canonical Command per Dimension + Final Aggregate Command

**What:** Define one command for D1, one for D2, and one final aggregate command (D1+D2+D3).
**Why:** Enables repeatable release validation and easy CI migration later.

### Pattern 3: Static Contract Guards for Streamlit API Hygiene

**What:** Keep D3 as source-level contract checks (no deprecated width keyword, no boolean width args, restricted literal policy).
**Why:** Fast, deterministic, and directly aligned with the regression vector introduced by Streamlit deprecations.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Final verification | Manual page clicking and ad hoc shell history | Canonical pytest/Make targets | Reproducible release evidence |
| D1 signal coverage | One giant brittle end-to-end test only | Focused phase gate + existing suite composition | Better failure localization |
| D3 guardrails | Runtime log scraping for deprecation warnings | AST/static keyword checks | Stable and fast under local/CI environments |

## Common Pitfalls

### Pitfall 1: Redundant tests without release structure

Adding more tests without defining canonical D1/D2/D3 commands still leaves release verification ambiguous.

### Pitfall 2: Over-coupled mega test fixtures

A single monolithic fixture spanning monitor, API, timeline, and Streamlit can be flaky and hard to debug. Keep dimension gates focused and aggregate at command level.

### Pitfall 3: D3 drift via narrow scope

Checking only one page for width regressions misses future reintroductions. Guard should cover runtime scope (`streamlit/app.py` + all `streamlit/pages/*.py`).

### Pitfall 4: Missing failure diagnostics

Release gates that fail without file/line/call details slow remediation and weaken confidence.

## Validation Architecture

### Validation Objective

Enforce D1/D2/D3 as a release contract with deterministic, repeatable evidence and clear failure diagnostics.

### Validation Layers

1. **D1 Gate (schema + monitor stability)**
   - Add `tests/autonomous/test_phase8_d1_monitor_schema_complete.py`
   - Compose with existing `test_agent_integration`, CLI monitor tests, API health/metrics schema readiness tests.

2. **D2 Gate (timeline data/render semantics)**
   - Add `tests/gui/test_phase8_d2_timeline_complete.py`
   - Compose with existing timeline unit tests + Phase 5 and 6 timeline gates.

3. **D3 Gate (Streamlit width guardrails)**
   - Reuse/expand `tests/gui/test_streamlit_width_compat.py` and `tests/gui/test_phase7_streamlit_compat_complete.py`
   - Add final release acceptance gate module for aggregated readiness.

4. **Aggregate Release Acceptance Command**
   - Add a canonical command/target running D1 + D2 + D3 suites together.

### Automated Gates

- `poetry run pytest tests/autonomous/test_phase8_d1_monitor_schema_complete.py tests/autonomous/test_agent_integration.py tests/test_cli.py tests/api/test_health.py tests/api/test_metrics.py -q -k "schema or monitor"`
- `poetry run pytest tests/gui/test_timeline.py tests/gui/test_complete_gui.py tests/gui/test_phase5_timeline_complete.py tests/gui/test_phase6_timeline_ux_complete.py tests/gui/test_phase8_d2_timeline_complete.py -q`
- `poetry run pytest tests/gui/test_streamlit_width_compat.py tests/gui/test_phase7_streamlit_compat_complete.py tests/gui/test_phase8_release_verification_complete.py -q`
- Final aggregate target: run all D1/D2/D3 gate commands in one invocation (`make test-release`).

## Code Evidence (Current State)

```python
# Existing D1 coverage (distributed)
# tests/autonomous/test_agent_integration.py
# tests/test_cli.py (monitor schema/runtime DB messaging)
# tests/api/test_health.py + tests/api/test_metrics.py (schema readiness)
```

```python
# Existing D2 coverage
# tests/gui/test_timeline.py
# tests/gui/test_phase5_timeline_complete.py
# tests/gui/test_phase6_timeline_ux_complete.py
```

```python
# Existing D3 coverage
# tests/gui/test_streamlit_width_compat.py
# tests/gui/test_phase7_streamlit_compat_complete.py
```

Coverage is strong but fragmented; Phase 8 should promote this into explicit release gates and canonical commands.

## Recommended Plan Split

1. **08-01 (Wave 1):** Build D1 release gate and consolidate schema/monitor regression command.
2. **08-02 (Wave 1):** Build D2 timeline release gate and consolidate timeline regression command.
3. **08-03 (Wave 2):** Finalize D3 guardrails and add aggregate release acceptance workflow (`test-release`).

## Sources

### Primary (HIGH confidence)
- `.planning/ROADMAP.md`
- `.planning/REQUIREMENTS.md`
- `.planning/STATE.md`
- `tests/autonomous/test_agent_integration.py`
- `tests/test_cli.py`
- `tests/api/test_health.py`
- `tests/api/test_metrics.py`
- `tests/gui/test_timeline.py`
- `tests/gui/test_complete_gui.py`
- `tests/gui/test_phase5_timeline_complete.py`
- `tests/gui/test_phase6_timeline_ux_complete.py`
- `tests/gui/test_streamlit_width_compat.py`
- `tests/gui/test_phase7_streamlit_compat_complete.py`

### Secondary (MEDIUM confidence)
- `Makefile`

## Metadata

**Confidence breakdown:**
- D1 regression architecture: HIGH
- D2 timeline gate strategy: HIGH
- D3 guard + release acceptance approach: HIGH

**Research date:** 2026-02-24
**Valid until:** 2026-03-24
