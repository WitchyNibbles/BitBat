# BitBat Reliability and Timeline Evolution

## Current State

- **Shipped version:** v1.2 (2026-02-26)
- **Milestone result:** Accuracy-evolution controls shipped with leakage-safe data/label
  contracts, cost-aware walk-forward evaluation, and promotion-gate deployment safety.
- **Release acceptance command:** `make test-release`

## Current Milestone: v1.3 Autonomous Monitor Alignment and Metrics Integrity

**Goal:** Ensure autonomous monitoring reports meaningful performance by aligning runtime
configuration with trained model artifacts and making no-data conditions explicit.

**Target features:**
- Enforce runtime/model alignment for `freq/horizon` so monitor cycles run against valid
  artifacts.
- Add fail-fast monitor diagnostics for missing model artifacts and mismatched runtime pairs.
- Improve monitor cycle/status observability so operators can distinguish "no predictions yet"
  from true zero performance.

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

### Active (v1.3 Planning Scope)

- [ ] ALGN-01: Monitor startup resolves the intended runtime config and uses the same
  `freq/horizon` pair as available model artifacts.
- [ ] ALGN-02: Monitor startup fails with actionable remediation when model artifacts for the
  resolved pair are missing.
- [ ] ALGN-03: Heartbeat and startup logs expose resolved config source and runtime pair.
- [ ] MON-04: Cycle output distinguishes "no prediction generated", "unrealized pending", and
  "realized metrics available" states.
- [ ] MON-05: Monitor status surfaces total/unrealized/realized counts for the active pair.
- [ ] MON-06: Missing-model root cause is visible in one operator-facing cycle summary.
- [ ] QUAL-07: Regression tests enforce startup guardrails for runtime/model mismatch.
- [ ] QUAL-08: Regression tests enforce non-ambiguous cycle/status metrics semantics.

### Out of Scope

- Advanced modeling expansion (microstructure, portfolio, EA optimization) before monitor
  alignment is corrected.
- Full dashboard redesign.
- Bypassing walk-forward and promotion-gate discipline for model experimentation.

## Context

On 2026-02-26, monitor cycles repeatedly logged all-zero metrics with
`drift_reason='Insufficient realized predictions: 0/30'`. Investigation showed runtime was
executing at `5m/30m` (heartbeat + logs) while local model artifacts and prediction history were
for `1h`-based pairs. This produced legitimate empty metrics, but with operator confusion.

v1.3 focuses on closing this operational gap before resuming advanced model exploration.

## Constraints

- **Compatibility:** Preserve existing autonomous DB schema and prior realized history.
- **Operational Safety:** Prefer fail-fast startup and explicit remediation over silent cycles.
- **Verification:** New monitor semantics must be covered by deterministic automated tests.
- **Scope:** Keep v1.3 focused on runtime alignment + observability, not model-family expansion.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Treat this as brownfield stabilization plus targeted enhancement (scope C) | Existing app surfaces were valuable but unstable in critical paths | ✓ Shipped in v1.0 |
| Done criteria are D1/D2/D3 | Technical acceptance gates needed to be explicit and verifiable | ✓ Enforced by canonical `make test-release` |
| Prioritize UI-first simplification for v1.1 | Operator value concentrated in five views; advanced views were broken | ✓ Shipped in v1.1 |
| Prioritize pipeline rigor over exotic model complexity for v1.2 | Leakage control, retraining cadence, and evaluation protocol dominate real-world crypto robustness | ✓ Shipped in v1.2 |
| Require promotion gates before autonomous deployment | Single-window wins are insufficient for stable production promotion | ✓ Shipped in v1.2 |
| Prioritize runtime alignment before advanced modeling in v1.3 | Monitoring trust is a prerequisite for evaluating any new model capabilities | — Pending |

## Next Milestone Goals

1. Eliminate silent monitor cycles caused by runtime/model pair mismatch.
2. Make cycle/status outputs self-explanatory when no realizations are available.
3. Lock behavior with regression tests before reopening advanced-model scope.

<details>
<summary>Archived deferred scope (post-v1.3 candidate)</summary>

- MICR-01: Optional microstructure/LOB feature pipeline.
- PORT-01: Multi-asset portfolio forecasting workflow.
- EA-01: EA-driven policy optimization under anti-overfitting controls.

</details>

---
*Last updated: 2026-02-26 after starting v1.3 milestone*
