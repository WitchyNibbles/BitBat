# BitBat Reliability and Timeline Evolution

## Current State

- **Shipped version:** v1.6 (2026-03-12)
- **Milestone result:** Live accuracy collapse diagnosed and corrected; reset + retrain recovery proven with fresh evidence (`239/300`, 79.67%); CLI/path/DB/XGBoost tech debt closed; milestone audit passed `10/10` requirements.
- **Archive references:** [v1.6 roadmap](/home/eimi/projects/ai-btc-predictor/.planning/milestones/v1.6-ROADMAP.md), [v1.6 requirements](/home/eimi/projects/ai-btc-predictor/.planning/milestones/v1.6-REQUIREMENTS.md), [v1.6 audit](/home/eimi/projects/ai-btc-predictor/.planning/milestones/v1.6-MILESTONE-AUDIT.md)

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
- ✓ UIF-01/02/03: Supported Streamlit surface constrained to Quick Start, Settings, Performance, About, and System.
- ✓ STAB-01/02/03: Runtime crash paths removed.
- ✓ RET-01/02: Retired-page UX and supported-surface guidance aligned.
- ✓ QUAL-04/05/06: Simplified UI contract, crash-signature guards, and supported-view smoke coverage release-wired.
- ✓ DATA-01/DATA-02/LABL-01: Leakage-safe as-of contracts and return-first/triple-barrier label modes shipped.
- ✓ MODL-01/MODL-02/MODL-03: Baseline families, rolling retraining windows, and regime/drift diagnostics shipped.
- ✓ EVAL-01/EVAL-02/EVAL-03/EVAL-04: Purge/embargo controls, cost-aware evaluation, champion reports, nested optimization safeguards shipped.
- ✓ OPER-02: Promotion requires consecutive incumbent outperformance and drawdown-safe gate pass.
- ✓ ALGN-01/02/03, SCHE-04, MON-04/05/06, QUAL-07/08/09: Runtime-pair guardrails, explicit cycle/status semantics, root-cause diagnostics, and monitor runbook/release gate hardening shipped in v1.3.
- ✓ APIC-01/02: API settings fallback to default.yaml and sub-hourly persistence — v1.4.
- ✓ SETT-01/02/03: React dashboard dynamic freq/horizon dropdowns with API-sourced defaults — v1.4.
- ✓ PRES-01/02/03: Scalper (5m/30m) and Swing (15m/1h) presets in Streamlit and React — v1.4.
- ✓ TEST-01/02: Preset parameter and settings round-trip test coverage — v1.4.
- ✓ AUDT-01/02/03/04/05, CORR-01/02/03/04/05/06, LEAK-01/02, ARCH-01/02/03/04/05/06 — v1.5.
- ✓ DIAG-01/02, FIXR-01/02/03/04, DEBT-01/02/03/04 — v1.6.

### Active

- Define the next milestone with `$gsd-new-milestone`.

### Deferred (Future Milestone Candidates)

- [ ] MICR-01: Add optional microstructure/LOB feature pipeline for short-horizon experiments.
- [ ] PORT-01: Add multi-asset portfolio-level forecasting and allocation workflow.
- [ ] EA-01: Add EA-driven policy optimization under strict anti-overfitting controls.

### Out of Scope

- Full dashboard redesign while monitor/runtime reliability remains the top operating requirement.
- Bypassing walk-forward and promotion-gate discipline for model experimentation.

## Context

As of 2026-03-12, v1.6 is shipped and archive-ready. The codebase carries:

- Fresh saved recovery evidence for the reset + retrain path
- A decomposed CLI package with archive-clean verification
- Config-driven model/metrics paths
- A unified runtime DB layer through `AutonomousDB`
- Classification-aligned XGBoost training, CV, and optimization semantics

## Next Milestone Goals

- Reassess product priorities after closing the accuracy-recovery and tech-debt track
- Define the next validated requirement set with `$gsd-new-milestone`

## Constraints

- **Compatibility:** Preserve existing autonomous DB schema and prior realized history.
- **Operational Safety:** Prefer fail-fast startup and explicit remediation over silent cycles.
- **Verification:** New monitor and modeling semantics must be covered by deterministic automated tests.
- **Scope Discipline:** New milestone work must preserve released runtime guarantees.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Treat this as brownfield stabilization plus targeted enhancement (scope C) | Existing app surfaces were valuable but unstable in critical paths | ✓ Shipped |
| Done criteria are explicit release gates | Technical acceptance gates needed to be explicit and verifiable | ✓ Enforced |
| Prioritize pipeline rigor over exotic model complexity | Leakage control, retraining cadence, and evaluation protocol dominate real-world robustness | ✓ Shipped |
| Prefer formal saved verification artifacts over inferred correctness | Audit closure required evidence that could survive milestone archival | ✓ Confirmed in v1.6 |

<details>
<summary>Archived v1.6 planning context</summary>

**Goal:** Diagnose and fix the live prediction accuracy collapse (~1%), restore the pipeline to a working state with clean reset + retrain, add an accuracy collapse guardrail to the monitor, and eliminate the four deferred tech debt items.

**Shipped:** Root-cause diagnosis, code fixes, reset CLI, monitor accuracy guardrail, CLI decomposition, path centralization, DB unification, XGBoost classification alignment, fresh recovery evidence closure, and CLI verification closure.

</details>

<details>
<summary>Archived v1.5 planning context</summary>

**Goal:** Comprehensive audit of the entire codebase against BitBat's core value promise — then fix critical gaps in pipeline correctness, architecture integrity, and production readiness.

**Shipped:** Complete audit baseline, all critical findings remediated, CI guardrails preventing recurrence.

</details>

*Last updated: 2026-03-12 after v1.6 milestone completion*
