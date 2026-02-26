# Project Retrospective

*A living document updated after each milestone. Lessons feed forward into future planning.*

## Milestone: v1.0 — Reliability and Timeline Evolution

**Shipped:** 2026-02-25
**Phases:** 9 | **Plans:** 24 | **Sessions:** 2

### What Was Built
- Runtime schema compatibility contracts, idempotent upgrades, and startup preflight enforcement.
- Monitor DB fault propagation and diagnostics across agent/CLI/API paths.
- Timeline reliability + UX expansion with filter controls and predicted-vs-realized analysis.
- Post-audit timeline readability closure with semantic marker grouping and opt-in comparison mode.
- Streamlit compatibility guardrails and release verification via `make test-release`.

### What Worked
- Strict phase sequencing (D1 → D2 → D3) reduced rework.
- Phase-level gate tests kept acceptance criteria explicit and repeatable, including gap-closure phases.

### What Was Inefficient
- Initial milestone closeout happened before running audit, requiring reopening and Phase 9 gap-closure work.
- Milestone closeout automation still required manual normalization of planning artifacts and milestone metadata.

### Patterns Established
- Keep each phase ending with explicit `*-SUMMARY.md` and phase-level verification artifacts.
- Use one canonical release command that chains requirement gates.
- Treat milestone audit as mandatory before final archival/tagging.

### Key Lessons
1. Enforcing canonical suite contracts in tests prevents silent release-process drift.
2. Planning docs need post-tool validation because automation can leave partial metadata.

### Cost Observations
- Model mix: quality profile, documentation-heavy closeout steps
- Sessions: 2
- Notable: most cost came from stabilization/testing; closeout required deliberate planning-state cleanup.

---

## Milestone: v1.1 — UI-First Simplification

**Shipped:** 2026-02-25
**Phases:** 3 | **Plans:** 8 | **Sessions:** 1

### What Was Built
- Reduced runtime UI surface to Quick Start, Settings, Performance, About, and System.
- Retired non-core pages from default discovery while preserving reference wrappers and shared retirement guidance.
- Hardened home rendering against missing `confidence` and partial prediction payloads.
- Locked retirement behavior and crash-path protections with dedicated Phase 10/11/12 regression suites.
- Added supported-view smoke coverage and required all v1.1 suites in `make test-release`.

### What Worked
- Sequencing surface pruning before runtime hardening avoided rework in legacy routes.
- Dedicated per-phase completion gate suites made release wiring explicit and quickly verifiable.

### What Was Inefficient
- Milestone closeout CLI counted full `.planning/phases` history, requiring manual correction of v1.1 milestone stats/accomplishments.
- Existing `.gitignore` behavior for new `.planning` archive files required explicit force-add handling in closeout flow.

### Patterns Established
- UI simplification milestones should include explicit retirement UX, not just hidden navigation.
- Every milestone-closing phase should wire new suites into both canonical D2 inventory and `make test-release`.

### Key Lessons
1. Retiring unstable surfaces with clear guidance is often lower risk than trying to patch legacy runtime paths in-place.
2. Milestone automation outputs should be validated against milestone scope before final commit/tag.

### Cost Observations
- Model mix: quality profile, test/verification-heavy execution
- Sessions: 1
- Notable: most effort was spent on regression-gate hardening and release-contract wiring, not feature expansion.

---

## Milestone: v1.2 — BTC Prediction Accuracy Evolution

**Shipped:** 2026-02-26
**Phases:** 4 | **Plans:** 12 | **Sessions:** 1

### What Was Built
- Leakage-safe as-of data and return-first label contracts, plus optional triple-barrier labeling mode.
- Dual baseline family training and rolling retraining/cv window controls with regime/drift diagnostics.
- Walk-forward evaluation hardening with purge/embargo leakage controls and fee/slippage net-vs-gross attribution.
- Deterministic candidate reports/champion decisions extended with nested optimization provenance.
- Multiple-testing safeguards and promotion-gate payloads enforced across CLI and autonomous retrainer deployment.

### What Worked
- Three-plan phase structure (data contracts → evaluation rigor → promotion safety) reduced dependency churn.
- Task-level atomic commits plus phase verification artifacts made late-stage milestone closeout low-risk.

### What Was Inefficient
- Milestone closeout CLI over-counted scope (all phase directories), requiring manual milestone-entry correction.
- No dedicated `v1.2-MILESTONE-AUDIT.md` was produced before archive/tag, reducing milestone-level verification traceability.

### Patterns Established
- Treat nested optimization provenance and safeguard payloads as first-class operational artifacts.
- Keep CLI and autonomous deployment on one shared champion/promotion decision schema.
- Run milestone audit before archival to avoid manual post-closeout caveats.

### Key Lessons
1. Promotion decisions need both statistical safeguards and sequence-based out-of-sample gate checks.
2. Milestone tooling output should be validated against intended scope before final commit/tag.

### Cost Observations
- Model mix: quality profile, implementation+verification balanced across one session
- Sessions: 1
- Notable: most effort was spent on contract wiring across optimize/evaluate/CLI/retrainer boundaries, not isolated model code.

---

## Cross-Milestone Trends

### Process Evolution

| Milestone | Sessions | Phases | Key Change |
|-----------|----------|--------|------------|
| v1.0 | 2 | 9 | Added post-audit gap-closure loop and milestone audit fail-gate enforcement |
| v1.1 | 1 | 3 | Established UI-surface retirement + runtime guard pattern with release-wired phase gates |
| v1.2 | 1 | 4 | Added nested optimization + safeguard/promotion-gate contracts shared by CLI and autonomous retrainer |

### Cumulative Quality

| Milestone | Tests | Coverage | Zero-Dep Additions |
|-----------|-------|----------|-------------------|
| v1.0 | D1/D2/D3 acceptance suites passing | Not tracked | Multiple test-only gating modules |
| v1.1 | D1/D2/D3 acceptance suites passing | Milestone requirements 11/11 satisfied | Phase10/11/12 regression + smoke gate suites |
| v1.2 | Phase 16 verification and targeted model/autonomous/CLI suites passing | Milestone requirements 11/11 satisfied | Nested optimize/safeguard/promotion gate regression contracts |

### Top Lessons (Verified Across Milestones)

1. Start each milestone with explicit acceptance gates and keep them executable as one command.
2. Treat planning-state updates as first-class deliverables, not optional docs cleanup.
3. Prefer safe retirement contracts over fragile legacy-path patching when operator value is concentrated elsewhere.
4. Close milestones only after scope-correct audit and archival stats validation.
