# Pitfalls Research

**Domain:** Brownfield monitor/database reliability and timeline UX modernization
**Researched:** 2026-02-24
**Confidence:** HIGH

## Critical Pitfalls

### Pitfall 1: ORM Model and Live DB Schema Drift

**What goes wrong:**
Code expects columns (`predicted_price`) that do not exist in existing SQLite tables.

**Why it happens:**
Schema changes are introduced in code without coordinated migration/version checks.

**How to avoid:**
Introduce explicit migration/version management and preflight schema compatibility checks at startup.

**Warning signs:**
- `OperationalError: no such column ...`
- Works on fresh DB but fails on long-lived local DBs

**Phase to address:**
Phase 1 (schema compatibility and migration safety).

---

### Pitfall 2: Timeline Logic Assumes Fully Realized/Complete Rows

**What goes wrong:**
Timeline charting fails or misleads when optional fields are null/missing or delayed.

**Why it happens:**
UI code tightly couples rendering assumptions to idealized DB rows.

**How to avoid:**
Use a timeline read-model adapter that normalizes missing values and explicitly models pending outcomes.

**Warning signs:**
- Empty/partial charts despite data present
- Runtime exceptions on null access or missing keys

**Phase to address:**
Phase 2-3 (timeline data contract + rendering hardening).

---

### Pitfall 3: Silent Exception Swallowing in Critical Paths

**What goes wrong:**
Monitor/UI appears alive while data freshness or correctness is broken.

**Why it happens:**
Broad `except Exception` with log-only or pass behavior in core loops.

**How to avoid:**
Differentiate critical vs non-critical failures; fail critical paths loudly and surface operator-visible diagnostics.

**Warning signs:**
- Missing updates without explicit failures
- Repeated warning logs with no actionable context

**Phase to address:**
Phase 4 (observability and failure semantics).

---

### Pitfall 4: Deprecation Debt Ignored Until Deadline

**What goes wrong:**
Widespread Streamlit warnings mask real issues and become urgent breaking changes near removal date.

**Why it happens:**
Incremental UI edits keep old APIs around.

**How to avoid:**
Execute one consistency sweep replacing all deprecated width arguments and add regression lint/tests.

**Warning signs:**
- Same warning repeated across multiple pages/buttons
- New code still using deprecated API

**Phase to address:**
Phase 3 (UI compatibility sweep).

---

### Pitfall 5: Fixes Without Regression Gates

**What goes wrong:**
Issues reappear after unrelated changes.

**Why it happens:**
No focused tests for schema compatibility, timeline contracts, or warning-free UI behavior.

**How to avoid:**
Add explicit regression tests and CI checks tied to D1/D2/D3 acceptance.

**Warning signs:**
- "Fixed" issues reopened repeatedly
- Manual-only verification of critical paths

**Phase to address:**
Phase 5+ (hardening and guardrails).

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Direct manual DB edits | Fast local unblock | Untracked schema drift | Never in shared workflow |
| Inline query/render logic in Streamlit pages | Quick feature addition | Hard-to-test timeline regressions | Only for throwaway prototypes |
| Broad catch-and-continue in monitor loop | Process keeps running | Data integrity and trust degradation | Only for non-critical telemetry branches |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| SQLAlchemy ↔ SQLite | Assuming columns/tables without migration | Schema preflight + versioned migration |
| Streamlit page widgets | Mixed deprecated/current width APIs | Standardize all components on `width=` |
| Predictor ↔ timeline | Inconsistent field naming/types | Central DTO/read-model transformation |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Full-table timeline queries | Slow page loads | Indexed queries + date window filtering | Larger prediction history windows |
| Recomputing expensive transforms per rerun | UI lag/jank | Cache stable transforms where safe | Frequent auto-refresh cycles |
| Excessive chart redraws | High CPU and visual flicker | Controlled refresh cadence + scoped updates | Busy operator sessions |

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| Exposing monitor/debug internals without controls | Operational data leakage | Restrict exposed endpoints and sanitize diagnostics |
| Keeping secrets in plain config defaults | Credential leakage | Environment/secret handling discipline |
| Treating local-only assumptions as permanent | Surprise exposure in shared deployment | Explicit deployment threat model in docs |

## UX Pitfalls

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| Timeline without realized-state cues | Users misread pending predictions as failures | Distinct pending vs realized visual semantics |
| Warnings visible during normal actions | Perceived instability | Zero-warning baseline for primary interactions |
| No actionable error guidance | Users blocked during incidents | Error banners with next-step remediation |

## "Looks Done But Isn't" Checklist

- [ ] **DB fix:** Migration works on existing long-lived DB, not just fresh DB.
- [ ] **Timeline fix:** Handles null/missing/pending rows and still renders correctly.
- [ ] **UI cleanup:** No remaining `use_container_width` in all dashboard pages/components.
- [ ] **Reliability claim:** Monitor run validated across multiple cycles without schema errors.
- [ ] **Regression safety:** Tests cover D1/D2/D3 behaviors.

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Schema drift outage | MEDIUM | Add/execute migration, backfill defaults, rerun monitor validation |
| Broken timeline rendering | LOW-MEDIUM | Normalize read model, patch renderer, add fixture-based test |
| Recurring deprecation warnings | LOW | Replace legacy API usage and enforce grep/test guard |

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| ORM/schema drift | Phase 1 | Migration + monitor startup integration tests pass |
| Timeline data assumptions | Phase 2-3 | Timeline tests pass with pending/realized mixed fixtures |
| Silent failure semantics | Phase 4 | Critical-path failures are surfaced and actionable |
| Deprecation debt | Phase 3 | Global search confirms no deprecated width API usage |
| Missing regression gates | Phase 5 | CI enforces D1/D2/D3 tests |

## Sources

- `.planning/PROJECT.md`
- `.planning/codebase/CONCERNS.md`
- `.planning/codebase/TESTING.md`
- `src/bitbat/autonomous/agent.py`, `models.py`, `db.py`
- `streamlit/` page modules

---
*Pitfalls research for: BitBat stabilization domain*
*Researched: 2026-02-24*
