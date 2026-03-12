# Milestones

## v1.6 Accuracy Recovery & Technical Debt Remediation (Shipped: 2026-03-12)

**Delivered:** Diagnosed and fixed the live accuracy collapse, proved fresh reset + retrain recovery, and closed all deferred v1.5 tech debt.

**Phases completed:** 29-37 (9 phases, 19 plans, 41 tasks)

**Key accomplishments:**
- Traced the live failure path with committed root-cause evidence before applying fixes.
- Fixed the classifier objective/inference/validator chain and added the reset + accuracy-guardrail operator path.
- Split the CLI monolith into focused command modules and re-verified the decomposed surface cleanly.
- Centralized artifact paths and unified runtime DB access behind `AutonomousDB`.
- Closed the live recovery evidence gap with `239/300` correct realized predictions (`79.67%`).

**Stats:**
- 110 files changed
- 34,276 lines of Python in the active codebase
- 9 phases, 19 plans, 41 tasks
- 4 days from milestone start to ship

**Git range:** `b58aec7^..f1d9285`

**What's next:** Define the next milestone with `$gsd-new-milestone`.

---

## v1.0 Reliability and Timeline Evolution (Shipped: 2026-02-25)

**Phases completed:** 9 phases, 24 plans, 72 tasks  
**Timeline:** 2026-02-24 13:27:13+01:00 to 2026-02-25 16:32:02+01:00  
**Git range:** `cd1d1ab^..07a6e86` (106 commits, 135 files changed, 11,181 insertions, 1,422 deletions)

**Key accomplishments:**
- Centralized schema compatibility contracts with additive migration safety and startup preflight diagnostics.
- Eliminated monitor DB runtime failure regressions with structured, operator-actionable fault handling.
- Aligned monitor persistence, validator realization semantics, and API/GUI prediction contracts.
- Stabilized timeline reliability with explicit pending vs realized semantics and sparse-price-safe rendering.
- Improved timeline readability under dense data and moved return comparison to explicit opt-in mode.
- Established release-grade D1/D2/D3 regression gates with canonical verification via `make test-release`.

**Verification snapshot:**
- Milestone audit passed (`v1.0-MILESTONE-AUDIT.md`): requirements 19/19, phases 9/9, integration 3/3, flows 3/3
- `make test-release` passed: D1 (21 passed), D2 (58 passed), D3 (11 passed)

---

## v1.1 UI-First Simplification (Shipped: 2026-02-25)

**Delivered:** Simplified BitBat to the five operator-used views and removed current runtime crash paths from retired advanced routes.

**Phases completed:** 10-12 (3 phases, 8 plans, 24 tasks)  
**Timeline:** 2026-02-25 17:07:18+01:00 to 2026-02-25 18:09:17+01:00  
**Git range:** `986678d^..0ca1cc9` (32 commits, 35 files changed, 1,582 insertions, 1,978 deletions)

**Key accomplishments:**
- Reduced Streamlit runtime navigation to `Quick Start`, `Settings`, `Performance`, `About`, and `System`, retiring non-core pages from default discovery.
- Hardened home prediction rendering to tolerate missing fields (including `confidence`) without `KeyError` crashes.
- Replaced legacy Backtest/Pipeline page entrypoints with retirement-safe notices that avoid brittle advanced imports.
- Added dedicated Phase 10/11/12 gate suites for supported-surface, runtime-stability, and simplified-UI regression contracts.
- Added supported-view smoke tests and wired all v1.1 gates into canonical D2 + `make test-release`.

**Verification snapshot:**
- Milestone audit passed (`v1.1-MILESTONE-AUDIT.md`): requirements 11/11, phases 3/3, integration 3/3, flows 3/3
- `make test-release` passed: D1 (21 passed), D2 (86 passed), D3 (13 passed)

---

## v1.2 BTC Prediction Accuracy Evolution (Shipped: 2026-02-26)

**Delivered:** Shipped leakage-safe data/label contracts, reproducible retraining/evaluation rigor, and promotion guardrails that block unstable candidates.

**Phases completed:** 13-16 (4 phases, 12 plans, 36 tasks)  
**Timeline:** 2026-02-26 00:17:46+01:00 to 2026-02-26 09:28:50+01:00  
**Git range:** `6de3d60^..ebd4a4d` (53 commits, 63 files changed, 7,028 insertions, 322 deletions)

**Key accomplishments:**
- Enforced as-of timestamp alignment and return-first label contracts to eliminate future leakage paths.
- Added optional triple-barrier labeling mode without breaking default return/direction dataset behavior.
- Standardized dual tree-ensemble baselines with rolling retraining windows and per-window regime/drift diagnostics.
- Added purge/embargo leakage controls and fee/slippage-aware net-vs-gross walk-forward evaluation outputs.
- Introduced nested walk-forward optimization with deterministic provenance and multiple-testing safeguard payloads.
- Enforced promotion gates across CLI and autonomous retraining (consecutive outperformance + drawdown guardrails) before deployment.

**Verification snapshot:**
- Phase 16 verification passed (`16-VERIFICATION.md`): 4/4 must-haves verified, EVAL-04 + OPER-02 complete.
- Milestone-level audit file (`v1.2-MILESTONE-AUDIT.md`) was not generated prior to closeout.

---

## v1.3 Autonomous Monitor Alignment and Metrics Integrity (Shipped: 2026-02-26)

**Delivered:** Locked monitor runtime/model alignment semantics and operator diagnostics so no-data states are explicit, test-gated, and runbooked for release operations.

**Phases completed:** 17-19 (3 phases, 8 plans, 24 tasks)  
**Timeline:** 2026-02-26 13:55:11+01:00 to 2026-02-26 17:03:35+01:00  
**Git range:** `c7afadc^..06671aa` (28 commits, 41 files changed, 2,909 insertions, 109 deletions)

**Key accomplishments:**
- Enforced monitor startup runtime-pair alignment with fail-fast model-artifact guardrails and actionable remediation guidance.
- Added config provenance to startup and heartbeat flows to expose resolved config source/path and runtime pair.
- Extended schema compatibility contracts to protect monitor status/snapshot/runtime paths from `performance_snapshots` drift.
- Added explicit cycle-state semantics (`prediction_state`, `prediction_reason`, `realization_state`) and concise root-cause diagnostics across agent, CLI, and heartbeat surfaces.
- Added phase-level D1 regression gates wired into canonical release verification so startup/cycle/schema regressions fail pre-release.
- Published a monitor operations runbook and service-template config contract (`--config`/`BITBAT_CONFIG`) enforced by documentation contract tests.

**Verification snapshot:**
- Phase 19 verification passed (`19-VERIFICATION.md`): 5/5 must-haves verified.
- `make test-release` passed after fixture normalization: D1 (36 passed), D2 (86 passed), D3/release-contract (13 passed).
- Milestone-level audit file (`v1.3-MILESTONE-AUDIT.md`) was not generated prior to closeout.

---

## v1.4 Configuration Alignment (Shipped: 2026-03-01)

**Phases completed:** 23 phases, 57 plans, 0 tasks

**Key accomplishments:**
- (none recorded)

---


## v1.5 Codebase Health Audit & Critical Remediation (Shipped: 2026-03-08)

**Phases completed:** 24 phases, 63 plans, 3 tasks

**Key accomplishments:**
- (none recorded)

---
