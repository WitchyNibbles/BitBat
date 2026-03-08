# BitBat Reliability and Timeline Evolution

## Current State

- **Shipped version:** v1.5 (2026-03-08)
- **Milestone result:** Full codebase audit completed; all silent production bugs fixed; architecture violations repaired; CI guardrails active; fold-aware OBV wired end-to-end. 638 tests passing.
- **Release acceptance command:** `poetry run pytest` (638 tests) + `poetry run ruff check src/ tests/` + `poetry run lint-imports`

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

- ✓ AUDT-01/02/03/04/05: Full codebase audit — test classification, dead code, branch coverage, complexity, E2E smoke test — v1.5.
- ✓ CORR-01/02/03/04/05/06: All silently broken production paths fixed — retrainer CLI contract, CV metric keys, regression_metrics purity, TypeError guards, test_leakage.py, API config defaults — v1.5.
- ✓ LEAK-01/02: OBV fold-boundary leakage assessed (2.33pp non-material) and fold-aware OBV implemented and wired into ContinuousTrainer — v1.5.
- ✓ ARCH-01/02/03/04/05/06: Private APIs promoted, price loading consolidated, config reset added, api→gui import eliminated, CI guardrails (ruff C901 + import-linter) active — v1.5.

### Active

- [ ] DIAG-01: Operator can identify which pipeline stage caused accuracy collapse
- [ ] DIAG-02: Root cause documented with reproducible trace before fix
- [ ] FIXR-01: Root cause of live accuracy ~1% fixed in code
- [ ] FIXR-02: Clean reset procedure (data/ + models/ + autonomous.db) executable and documented
- [ ] FIXR-03: After reset + retrain, live directional accuracy exceeds 33%
- [ ] FIXR-04: Monitor alerts when realized accuracy falls below configurable threshold
- [ ] DEBT-01: CLI monolith decomposed (cli.py 1802+ lines, 53 functions → focused modules)
- [ ] DEBT-02: Hardcoded Path("models")/Path("metrics") centralized (15+ sites)
- [ ] DEBT-03: Dual DB access unified (SQLAlchemy ORM + raw sqlite3 → single approach)
- [ ] DEBT-04: XGBoost objective mismatch fixed (reg:squarederror → classification objective)

### Deferred (Future Milestone Candidates)

- [ ] MICR-01: Add optional microstructure/LOB feature pipeline for short-horizon experiments.
- [ ] PORT-01: Add multi-asset portfolio-level forecasting and allocation workflow.
- [ ] EA-01: Add EA-driven policy optimization under strict anti-overfitting controls.

### Out of Scope

- Full dashboard redesign while monitor/runtime reliability remains the top operating requirement.
- Bypassing walk-forward and promotion-gate discipline for model experimentation.

## Context

As of 2026-03-08, v1.5 completed the codebase health audit and remediation milestone. Key outcomes:
- **638 tests passing** with pytest markers on all 76 test files (behavioral/integration/structural)
- **19/19 requirements satisfied** — all silent production bugs fixed, architecture violations repaired, CI guardrails active
- **Remaining tech debt** explicitly deferred to v1.6+: cli.py monolith, path hardcoding, dual DB, XGBoost objective mismatch
- **CI gates** now enforce: ruff C901 complexity (max=10) + import-linter forbidden contract (api→gui)
- ~30,381 Python LOC across src/ + tests/

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

## Current Milestone: v1.6 Accuracy Recovery & Technical Debt Remediation

**Goal:** Diagnose and fix the live prediction accuracy collapse (~1%), restore the pipeline to a working state with clean reset + retrain if needed, and eliminate the four deferred tech debt items.

**Target features:**
- Root-cause diagnosis of live accuracy collapse
- Clean reset + retrain procedure
- Accuracy collapse guardrail in monitor
- CLI monolith decomposition (DEBT-01)
- Hardcoded path centralization (DEBT-02)
- Dual DB access unification (DEBT-03)
- XGBoost objective mismatch fix (DEBT-04)

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

| Audit before adding features | 4 milestones shipped rapidly; tech debt audit before expanding scope reduces compounding risk | ✓ Confirmed — v1.5 audit found 14 significant issues, all critical ones fixed |
| 11 noqa:C901 suppressions on pre-existing complexity | Enabling C901 gate required suppressing legacy violations rather than fixing them all; cleanest path forward without scope creep | ✓ Good — suppressions documented, gate active, DEBT-01 tracks the backlog |
| fold_boundaries wired in ContinuousTrainer only (not CV loop) | Inference paths intentionally use fold_boundaries=None; OBV leakage non-material (2.33pp); retraining path is the correct activation site | ✓ Good — behavioral test confirms contract |

---

<details>
<summary>Archived v1.5 planning context</summary>

**Goal:** Comprehensive audit of the entire codebase against BitBat's core value promise — then fix critical gaps in pipeline correctness, architecture integrity, and production readiness.

**Shipped:** Complete audit baseline (vulture, radon, coverage, markers, smoke test), all 14 critical/high/medium findings remediated, CI guardrails preventing recurrence.

</details>

*Last updated: 2026-03-08 after v1.6 milestone start*
