# BitBat Reliability and Timeline Evolution

## Current State

- **Shipped version:** v1.1 (2026-02-25)
- **Milestone result:** UI-first simplification shipped with a five-view supported surface, retirement-safe legacy routes, and regression/smoke release gates.
- **Release acceptance command:** `make test-release`

## Current Milestone: v1.2 BTC Prediction Accuracy Evolution

**Goal:** Improve BTC prediction accuracy by upgrading data/label quality, retraining/evaluation rigor, and model promotion safety.

**Target features:**
- Return-first and leakage-safe training data contracts (with optional triple-barrier event labels for trading-aligned experiments).
- Strong classical baselines (tree ensembles) with periodic retraining and regime/drift diagnostics.
- Walk-forward, cost-aware evaluation and explicit promotion gates that prevent backtest overfitting.

## What This Is

BitBat is a local-first BTC prediction platform with CLI, API, autonomous monitoring, and a Streamlit dashboard. After v1.1, the primary operator UI is intentionally reduced to reliable core views with retired advanced paths guarded behind user-facing guidance.

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

### Active (Next Milestone Planning Queue)

- [ ] Upgrade the prediction dataset to leakage-safe as-of semantics and return-based targets.
- [ ] Add optional triple-barrier labels for event-driven signal evaluation.
- [ ] Establish tree-ensemble accuracy baselines with reproducible retraining windows.
- [ ] Introduce cost-aware walk-forward evaluation with statistical guardrails for model selection.
- [ ] Promote models only through explicit multi-window performance and drawdown gates.

### Out of Scope

- Full dashboard redesign.
- Major model strategy replacement.
- Reintroducing advanced views without explicit operator demand and verification contracts.
- Treating raw next-price forecasting as the primary objective instead of return/direction/trading-aligned targets.

## Context

v1.1 completed in 3 phases (10-12), 8 plans, and 24 tasks. The runtime surface is now intentionally minimal, retired advanced routes fail safely, and release acceptance includes dedicated phase-level coverage for supported-surface, stability, and smoke behavior.

The next milestone is grounded in `deep-research-report.md` findings: crypto prediction accuracy gains come more from leakage control, retraining discipline, and cost-aware walk-forward evaluation than from jumping directly to complex model architectures.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Treat this as brownfield stabilization plus targeted enhancement (scope C) | Existing app surfaces were valuable but unstable in critical paths | ✓ Shipped in v1.0 |
| Done criteria are D1/D2/D3 | Technical acceptance gates needed to be explicit and verifiable | ✓ Enforced by canonical `make test-release` |
| Prioritize UI-first simplification for v1.1 | Operator value concentrated in five views; advanced views were broken | ✓ Shipped in v1.1 |
| Retire Backtest/Pipeline routes instead of patching brittle imports in place | Fastest safe path to remove user-facing tracebacks while preserving future rebuild option | ✓ Shipped in v1.1 |
| Lock milestone behavior with dedicated phase gates plus release wiring | Prevent regressions when future work touches Streamlit/runtime contracts | ✓ Shipped in v1.1 |
| Prioritize pipeline rigor over exotic model complexity for v1.2 | Research review indicates leakage, retraining cadence, and evaluation protocol dominate real-world crypto accuracy | — Pending |

## Next Milestone Goals

1. Shift prediction contracts to return/direction-focused labels with strict as-of leakage controls.
2. Build a reproducible retraining and walk-forward benchmark harness that includes transaction-cost realism.
3. Add statistically safer promotion gates so “improved” models must beat incumbents out-of-sample before shipping.

<details>
<summary>Archived v1.1 kickoff notes</summary>

- Goal: reduce BitBat to actively used views and retire broken surfaces safely.
- Targeted runtime failures: missing `confidence`, pipeline import failure, backtest indexing crash.

</details>

---
*Last updated: 2026-02-25 after starting v1.2 milestone*
