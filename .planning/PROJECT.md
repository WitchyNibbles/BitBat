# BitBat Reliability and Timeline Evolution

## Current State

- **Shipped version:** v1.3 (2026-02-26)
- **Milestone result:** Monitor runtime/model alignment and diagnostics integrity are now release-gated and operator-documented.
- **Release acceptance command:** `make test-release`

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

### Active (Next Milestone Candidates)

- [ ] MICR-01: Add optional microstructure/LOB feature pipeline for short-horizon experiments.
- [ ] PORT-01: Add multi-asset portfolio-level forecasting and allocation workflow.
- [ ] EA-01: Add EA-driven policy optimization under strict anti-overfitting controls.

### Out of Scope

- Full dashboard redesign while monitor/runtime reliability remains the top operating requirement.
- Bypassing walk-forward and promotion-gate discipline for model experimentation.

## Context

As of 2026-02-26, v1.3 closed the monitor trust gap: startup now validates runtime/model pairing,
no-prediction states are explicit, and release verification includes phase-level alignment gates
plus runbook contract tests.

## Constraints

- **Compatibility:** Preserve existing autonomous DB schema and prior realized history.
- **Operational Safety:** Prefer fail-fast startup and explicit remediation over silent cycles.
- **Verification:** New monitor and modeling semantics must be covered by deterministic automated
  tests.
- **Scope Discipline:** New milestone work must preserve released v1.3 monitor guarantees.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Treat this as brownfield stabilization plus targeted enhancement (scope C) | Existing app surfaces were valuable but unstable in critical paths | ✓ Shipped in v1.0 |
| Done criteria are D1/D2/D3 | Technical acceptance gates needed to be explicit and verifiable | ✓ Enforced by canonical `make test-release` |
| Prioritize UI-first simplification for v1.1 | Operator value concentrated in five views; advanced views were broken | ✓ Shipped in v1.1 |
| Prioritize pipeline rigor over exotic model complexity for v1.2 | Leakage control, retraining cadence, and evaluation protocol dominate real-world crypto robustness | ✓ Shipped in v1.2 |
| Require promotion gates before autonomous deployment | Single-window wins are insufficient for stable production promotion | ✓ Shipped in v1.2 |
| Prioritize runtime alignment before advanced modeling in v1.3 | Monitoring trust is a prerequisite for evaluating new model capabilities | ✓ Shipped in v1.3 |

## Next Milestone Goals

1. Reopen advanced-model scope only on top of v1.3 monitor alignment and diagnostics guarantees.
2. Add new model capabilities with reproducible, leakage-safe, cost-aware evaluation evidence.
3. Extend release verification contracts as scope expands so regressions are caught pre-release.

<details>
<summary>Archived v1.3 planning context</summary>

**Goal:** Ensure autonomous monitoring reports meaningful performance by aligning runtime
configuration with trained model artifacts and making no-data conditions explicit.

**Target features:**
- Enforce runtime/model alignment for `freq/horizon` so monitor cycles run against valid
  artifacts.
- Add fail-fast monitor diagnostics for missing model artifacts and mismatched runtime pairs.
- Improve monitor cycle/status observability so operators can distinguish "no predictions yet"
  from true zero performance.

</details>

---
*Last updated: 2026-02-26 after completing v1.3 milestone*
