# BitBat Reliability and Timeline Evolution

## Current Milestone: v1.5 Codebase Health Audit & Critical Remediation

**Goal:** Comprehensive audit of the entire codebase against BitBat's core value promise — then fix critical gaps in pipeline correctness, architecture integrity, and production readiness.

**Target features:**
- Full codebase audit across pipeline correctness, architecture drift, dead/broken code
- End-to-end usability validation (clone → ingest → predict → monitor)
- Production readiness assessment (maintainability, deployability, error handling)
- Critical issue remediation (high-severity fixes shipped; lower-severity cataloged)

## Current State

- **Shipped version:** v1.4 (2026-03-01)
- **Milestone result:** UI settings, presets, and API defaults now reflect the actual runtime configuration across the full sub-hourly frequency range.
- **Release acceptance command:** `make test-release` (169 tests across 4 gates)

## What This Is

BitBat is a local-first BTC prediction platform with CLI, API, autonomous monitoring, and a
Streamlit dashboard. The runtime combines a simplified operator UI with deterministic,
audit-friendly model evaluation and promotion controls.

## Core Value

A reliable prediction system where operators can trust that monitoring outputs correspond to real,
active prediction flows for the configured runtime pair.

## Requirements

### Validated

- ✓ SCHE-01/02/03: Runtime schema compatibility, startup preflight, and idempotent upgrades.
- ✓ MON-01/02/03: Monitor runtime fault handling and diagnostics.
- ✓ TIM-01/02/03/04/05: Timeline reliability, UX expansion, readability, and comparison clarity.
- ✓ GUI-01/02/03: Streamlit compatibility and width-contract hardening.
- ✓ API-01/02: API/readiness alignment with runtime semantics.
- ✓ QUAL-01/02/03: D1/D2/D3 release verification contracts.
- ✓ UIF-01/02/03: Supported Streamlit surface constrained to Quick Start, Settings, Performance,
  About, and System.
- ✓ STAB-01/02/03: Runtime crash paths removed (`confidence`, pipeline import, backtest
  indexing).
- ✓ RET-01/02: Retired-page UX and supported-surface guidance aligned.
- ✓ QUAL-04/05/06: Simplified UI contract, crash-signature guards, and supported-view smoke
  coverage release-wired.
- ✓ DATA-01/DATA-02/LABL-01: Leakage-safe as-of contracts and return-first/triple-barrier label
  modes shipped.
- ✓ MODL-01/MODL-02/MODL-03: Baseline families, rolling retraining windows, and regime/drift
  diagnostics shipped.
- ✓ EVAL-01/EVAL-02/EVAL-03/EVAL-04: Purge/embargo controls, cost-aware evaluation, champion
  reports, nested optimization safeguards shipped.
- ✓ OPER-02: Promotion requires consecutive incumbent outperformance and drawdown-safe gate pass.
- ✓ ALGN-01/02/03, SCHE-04, MON-04/05/06, QUAL-07/08/09: Runtime-pair guardrails, explicit
  cycle/status semantics, root-cause diagnostics, and monitor runbook/release gate hardening
  shipped in v1.3.
- ✓ APIC-01/02: API settings fallback to default.yaml and sub-hourly persistence — v1.4.
- ✓ SETT-01/02/03: React dashboard dynamic freq/horizon dropdowns with API-sourced defaults — v1.4.
- ✓ PRES-01/02/03: Scalper (5m/30m) and Swing (15m/1h) presets in Streamlit and React — v1.4.
- ✓ TEST-01/02: Preset parameter and settings round-trip test coverage — v1.4.

### Active

- [ ] AUDIT-01: Comprehensive codebase audit for incongruences, code smells, and design errors
- [ ] AUDIT-02: End-to-end pipeline usability validation (ingestion through monitoring)
- [ ] AUDIT-03: Production readiness assessment (error handling, maintainability, deployability)
- [ ] AUDIT-04: Critical issue remediation for high-severity findings

### Deferred (Future Milestone Candidates)

- [ ] MICR-01: Add optional microstructure/LOB feature pipeline for short-horizon experiments.
- [ ] PORT-01: Add multi-asset portfolio-level forecasting and allocation workflow.
- [ ] EA-01: Add EA-driven policy optimization under strict anti-overfitting controls.

### Out of Scope

- Full dashboard redesign while monitor/runtime reliability remains the top operating requirement.
- Bypassing walk-forward and promotion-gate discipline for model experimentation.

## Context

As of 2026-03-01, v1.4 closed the configuration alignment gap: the API, React dashboard, and
Streamlit GUI all expose the full sub-hourly frequency range (5m, 15m, 30m) with correct defaults
from default.yaml, named trading presets (Scalper, Swing), human-readable labels, and automated
regression tests covering both preset values and API round-trip persistence.

## Constraints

- **Compatibility:** Preserve existing autonomous DB schema and prior realized history.
- **Operational Safety:** Prefer fail-fast startup and explicit remediation over silent cycles.
- **Verification:** New monitor and modeling semantics must be covered by deterministic automated
  tests.
- **Scope Discipline:** New milestone work must preserve released v1.3 monitor guarantees and v1.4 configuration contracts.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Treat this as brownfield stabilization plus targeted enhancement (scope C) | Existing app surfaces were valuable but unstable in critical paths | ✓ Shipped in v1.0 |
| Done criteria are D1/D2/D3 | Technical acceptance gates needed to be explicit and verifiable | ✓ Enforced by canonical `make test-release` |
| Prioritize UI-first simplification for v1.1 | Operator value concentrated in five views; advanced views were broken | ✓ Shipped in v1.1 |
| Prioritize pipeline rigor over exotic model complexity for v1.2 | Leakage control, retraining cadence, and evaluation protocol dominate real-world crypto robustness | ✓ Shipped in v1.2 |
| Require promotion gates before autonomous deployment | Single-window wins are insufficient for stable production promotion | ✓ Shipped in v1.2 |
| Prioritize runtime alignment before advanced modeling in v1.3 | Monitoring trust is a prerequisite for evaluating new model capabilities | ✓ Shipped in v1.3 |
| Align UI/API config with default.yaml reality for v1.4 | Operators saw 1h-only options while runtime used 5m; trust requires visible truth | ✓ Shipped in v1.4 |
| API as single source of truth for config values | Eliminates hardcoded frontend defaults diverging from backend | ✓ Shipped in v1.4 |

## Current Focus

Milestone v1.5: Codebase Health Audit & Critical Remediation — auditing the full codebase for
incongruences, code smells, design errors, and broken paths against the core value promise, then
fixing critical issues.

<details>
<summary>Archived v1.4 planning context</summary>

**Goal:** Make the UI settings, presets, and API defaults reflect the actual runtime configuration (5m freq) instead of hardcoded 1h-only options.

**Shipped:** API settings fallback, dynamic React dropdowns, Scalper/Swing presets, human-readable
labels, automated test coverage for presets and round-trip persistence.

</details>

<details>
<summary>Archived v1.3 planning context</summary>

**Goal:** Ensure autonomous monitoring reports meaningful performance by aligning runtime
configuration with trained model artifacts and making no-data conditions explicit.

**Shipped:** Runtime-pair guardrails, explicit cycle/status semantics, root-cause diagnostics,
monitor runbook and release gate hardening.

</details>

| Audit before adding features | 4 milestones shipped rapidly; tech debt audit before expanding scope reduces compounding risk | — Pending |

---
*Last updated: 2026-03-04 after v1.5 milestone start*
