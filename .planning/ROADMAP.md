# Roadmap: BitBat Reliability and Timeline Evolution

## Milestones

- ✅ **v1.0 Reliability and Timeline Evolution** — Phases 1-8 shipped on 2026-02-24 ([roadmap archive](milestones/v1.0-ROADMAP.md), [requirements archive](milestones/v1.0-REQUIREMENTS.md)).

## Active Gap Closure

- ⚠️ **Post-audit closure (2026-02-25)** — Milestone audit found timeline readability/comprehension gaps requiring follow-up work before considering v1.0 fully complete.

### Phase 9: Timeline Readability and Overlay Clarity
**Goal:** Restore at-a-glance interpretability of the prediction timeline while preserving overlay comparison value.
**Depends on:** Phase 8 verification baseline
**Requirements:** [TIM-03, TIM-05]
**Gap Closure:** Closes gaps from `.planning/v1.0-MILESTONE-AUDIT.md` (requirements, integration, and operator flow interpretation).

Planned tasks:
- Rework timeline visual hierarchy so direction/confidence cues are readable under dense data.
- Adjust default overlay behavior and controls to reduce first-view cognitive load.
- Simplify predicted-vs-realized overlay composition and mismatch rendering semantics.
- Add regression checks for timeline readability/comprehension acceptance criteria.

## Next

- Plan and execute Phase 9.
- Re-run `$gsd-audit-milestone` after Phase 9 verification.
