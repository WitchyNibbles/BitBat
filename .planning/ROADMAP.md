# Roadmap: BitBat Reliability and Timeline Evolution

## Milestones

- ✅ **v1.0 Reliability and Timeline Evolution** — Phases 1-9 shipped on 2026-02-25 ([roadmap archive](milestones/v1.0-ROADMAP.md), [requirements archive](milestones/v1.0-REQUIREMENTS.md), [audit archive](milestones/v1.0-MILESTONE-AUDIT.md)).
- ✅ **v1.1 UI-First Simplification** — Phases 10-12 shipped on 2026-02-25 ([roadmap archive](milestones/v1.1-ROADMAP.md), [requirements archive](milestones/v1.1-REQUIREMENTS.md), [audit archive](milestones/v1.1-MILESTONE-AUDIT.md)).
- ✅ **v1.2 BTC Prediction Accuracy Evolution** — Phases 13-16 shipped on 2026-02-26 ([roadmap archive](milestones/v1.2-ROADMAP.md), [requirements archive](milestones/v1.2-REQUIREMENTS.md)).
- 🚧 **v1.3 Autonomous Monitor Alignment and Metrics Integrity** — Phases 17-19 planned (eliminate runtime/model mismatch loops and make monitor no-data states explicit).

## v1.3 Planned Phases

### Phase 17: Runtime Pair Alignment and Startup Guardrails
**Goal:** Ensure monitor startup uses an intended runtime pair and blocks immediately on missing model artifacts.
**Depends on:** v1.2 verified baseline
**Requirements:** [ALGN-01, ALGN-02, ALGN-03, SCHE-04]
**Plans:** 3/3 plans complete
**Status:** Complete (2026-02-26)

Success criteria:
1. Monitor startup reports resolved config source and resolved `freq/horizon` before cycle execution.
2. Startup fails fast with actionable remediation when `models/<freq>_<horizon>/xgb.json` is missing.
3. Heartbeat includes config path/source metadata in addition to `freq` and `horizon`.
4. Schema compatibility contract covers `performance_snapshots` runtime columns needed by monitor status/snapshots.
5. Regression coverage verifies mismatch scenarios cannot enter silent monitoring loops.

### Phase 18: Monitoring Cycle Semantics and Operator Diagnostics
**Goal:** Make monitor outputs unambiguous when predictions are missing, pending realization, or realized.
**Depends on:** Phase 17
**Requirements:** [MON-04, MON-05, MON-06]
**Plans:** 3/3 plans complete
**Status:** Complete (2026-02-26)

Success criteria:
1. Cycle summary includes explicit status fields for prediction generation, pending validations, and realized sample availability.
2. `bitbat monitor status` surfaces total/unrealized/realized counts for the active pair.
3. Missing-model root cause appears in operator-facing logs without needing traceback inspection.
4. Performance snapshot behavior remains consistent and documented when realized count is zero.

### Phase 19: Regression Gates and Runbook Hardening
**Goal:** Lock v1.3 behavior with tests and operational documentation so misconfiguration regressions are caught pre-release.
**Depends on:** Phase 18
**Requirements:** [QUAL-07, QUAL-08, QUAL-09]
**Plans:** 0/2 plans complete
**Status:** Pending

Success criteria:
1. Automated tests fail if startup guardrails for runtime/model mismatch are bypassed.
2. Automated tests fail if cycle/status payloads regress to ambiguous all-zero semantics.
3. Automated tests fail when schema compatibility omits runtime-required `performance_snapshots` columns.
4. Deployment/runbook docs include supported config wiring (`--config` or `BITBAT_CONFIG`) for monitor services.
5. Canonical release verification includes v1.3 monitor alignment/diagnostic regressions.

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 17. Runtime Pair Alignment and Startup Guardrails | v1.3 | 3/3 | Complete | 2026-02-26 |
| 18. Monitoring Cycle Semantics and Operator Diagnostics | v1.3 | 3/3 | Complete | 2026-02-26 |
| 19. Regression Gates and Runbook Hardening | v1.3 | 0/2 | Pending | — |

## Next

- Start with `$gsd-discuss-phase 19` to confirm regression gate and runbook scope.
- Then run `$gsd-plan-phase 19` to generate executable plans.
