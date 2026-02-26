# BitBat Reliability and Timeline Evolution

## Current State

- **Shipped version:** v1.2 (2026-02-26)
- **Milestone result:** Accuracy-evolution controls shipped with leakage-safe data/label contracts, cost-aware walk-forward evaluation, and promotion-gate deployment safety.
- **Release acceptance command:** `make test-release`

## Next Milestone Candidate: v1.3 Advanced Modeling Exploration

**Goal:** Extend v1.2’s robust baseline pipeline with carefully scoped advanced modeling experiments.

**Target features:**
- Optional microstructure/LOB feature experiments behind existing leakage-safe contracts.
- Multi-asset or portfolio-aware forecasting experiments that reuse current evaluation/promotion gates.
- EA-driven policy or threshold optimization under strict anti-overfitting controls.

## What This Is

BitBat is a local-first BTC prediction platform with CLI, API, autonomous monitoring, and a Streamlit dashboard. The runtime now combines a simplified operator UI with deterministic, audit-friendly model evaluation and promotion controls.

## Core Value

A reliable prediction system where operators can trust that monitoring runs without DB failures and the timeline shows clear prediction vs. outcome history.

## Requirements

### Validated

- ✓ SCHE-01/02/03: Runtime schema compatibility, startup preflight, and idempotent upgrades.
- ✓ MON-01/02/03: Monitor runtime fault handling and diagnostics.
- ✓ TIM-01/02/03/04/05: Timeline reliability, UX expansion, readability, and comparison clarity.
- ✓ GUI-01/02/03: Streamlit compatibility and width-contract hardening.
- ✓ API-01/02: API/readiness alignment with runtime semantics.
- ✓ QUAL-01/02/03: D1/D2/D3 release verification contracts.
- ✓ UIF-01/02/03: Supported Streamlit surface constrained to Quick Start, Settings, Performance, About, and System.
- ✓ STAB-01/02/03: Runtime crash paths removed (`confidence`, pipeline import, backtest indexing).
- ✓ RET-01/02: Retired-page UX and supported-surface guidance aligned.
- ✓ QUAL-04/05/06: Simplified UI contract, crash-signature guards, and supported-view smoke coverage release-wired.
- ✓ DATA-01/DATA-02/LABL-01: Leakage-safe as-of contracts and return-first/triple-barrier label modes shipped.
- ✓ MODL-01/MODL-02/MODL-03: Baseline families, rolling retraining windows, and regime/drift diagnostics shipped.
- ✓ EVAL-01/EVAL-02/EVAL-03/EVAL-04: Purge/embargo controls, cost-aware evaluation, champion reports, nested optimization safeguards shipped.
- ✓ OPER-02: Promotion requires consecutive incumbent outperformance and drawdown-safe gate pass.

### Active (Next Milestone Planning Queue)

- [ ] MICR-01: Add optional microstructure/LOB feature pipeline for short-horizon signal experiments.
- [ ] PORT-01: Add multi-asset portfolio-level forecasting and allocation workflow.
- [ ] EA-01: Add EA-driven policy optimization under strict anti-overfitting controls.

### Out of Scope

- Full dashboard redesign.
- Reintroducing advanced UI views without explicit operator demand and verification contracts.
- Bypassing walk-forward and promotion-gate discipline for model experimentation.

## Context

v1.2 completed in 4 phases (13-16), 12 plans, and 36 tasks. The system now enforces deterministic nested optimization provenance, multiple-testing safeguards, and promotion-gate vetoes across CLI evaluation and autonomous retraining.

Next milestone work should build on this safety baseline instead of bypassing it; new model complexity must remain subordinate to reproducible out-of-sample evidence.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Treat this as brownfield stabilization plus targeted enhancement (scope C) | Existing app surfaces were valuable but unstable in critical paths | ✓ Shipped in v1.0 |
| Done criteria are D1/D2/D3 | Technical acceptance gates needed to be explicit and verifiable | ✓ Enforced by canonical `make test-release` |
| Prioritize UI-first simplification for v1.1 | Operator value concentrated in five views; advanced views were broken | ✓ Shipped in v1.1 |
| Prioritize pipeline rigor over exotic model complexity for v1.2 | Leakage control, retraining cadence, and evaluation protocol dominate real-world crypto robustness | ✓ Shipped in v1.2 |
| Require promotion gates before autonomous deployment | Single-window wins are insufficient for stable production promotion | ✓ Shipped in v1.2 |

## Next Milestone Goals

1. Define v1.3 requirements/roadmap with explicit acceptance gates before implementation starts.
2. Evaluate advanced feature/model candidates (MICR-01/PORT-01/EA-01) within existing leakage-safe and cost-aware contracts.
3. Preserve deterministic artifacts and auditability for all new optimization and promotion paths.

<details>
<summary>Archived v1.2 kickoff notes</summary>

- Goal: improve BTC prediction accuracy via pipeline rigor before model-complexity expansion.
- Scope anchors: DATA-01/02, LABL-01, MODL-01/02/03, EVAL-01/02/03/04, OPER-02.

</details>

---
*Last updated: 2026-02-26 after completing v1.2 milestone*
