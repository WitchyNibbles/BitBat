# Phase 2: Migration Safety & Startup Readiness - Research

**Researched:** 2026-02-24
**Domain:** Idempotent schema migration hardening and readiness signal accuracy
**Confidence:** HIGH

## Summary

Phase 1 established a compatibility contract and additive upgrade path, but readiness surfaces still rely heavily on file existence checks and broad exception fallbacks. For Phase 2, the primary risk is false readiness: endpoints may report "database available" while schema compatibility is degraded.

This phase should harden migration idempotency semantics and expose schema compatibility state explicitly in API health/status outputs. The migration path must remain additive and non-destructive while becoming observably safe across repeated runs.

**Primary recommendation:** build a shared readiness probe that uses `audit_schema_compatibility` (non-mutating) and wire it into `/health/detailed`, `/analytics/status`, and related monitoring surfaces.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| SCHE-03 | Schema upgrade path is idempotent and preserves existing prediction history | Extend repeat-run tests across script/runtime flows; maintain additive-only migration behavior |
| API-02 | Health/status surfaces reflect schema incompatibility accurately | Replace simple file-exists checks with schema-aware readiness status and actionable details |
</phase_requirements>

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| SQLAlchemy | 2.x | Schema introspection, migration, and DB sessions | Existing migration/runtime backbone in autonomous stack |
| FastAPI | current repo baseline | Health/readiness endpoint delivery | Existing API surface for operator status |
| pytest + TestClient | current repo baseline | Regression and readiness behavior tests | Existing test pattern for routes and autonomous DB |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `schema_compat` module | local | Audit/upgrade contract source of truth | Any migration or readiness check touching runtime schema assumptions |
| CLI script (`init_autonomous_db.py`) | local | Operator migration workflows | Audit/upgrade/force paths with explicit status messaging |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|-----------|-----------|----------|
| Schema-aware readiness checks | Continue file-presence checks only | Simpler, but produces false "ready" signals when schema is incompatible |
| Additive idempotent migration | Table recreation as default fix | Faster reset but destroys historical prediction rows |

## Architecture Patterns

### Recommended Project Structure

```text
src/bitbat/autonomous/
├── schema_compat.py            # Contract + audit + upgrade
├── db.py                       # Runtime init + compatibility enforcement

src/bitbat/api/
├── schemas.py                  # Readiness response fields
└── routes/
    ├── health.py               # Detailed readiness reporting
    └── analytics.py            # Status endpoint enriched with schema compatibility

tests/
├── autonomous/test_schema_compat.py
├── autonomous/test_init_script.py
└── api/test_health.py
```

### Pattern 1: Audit-First Readiness

**What:** Use non-mutating compatibility audit for health/status checks.
**When to use:** API readiness endpoints and status reports.
**Example:**

```python
report = audit_schema_compatibility(engine=engine)
compatible = report.is_compatible
missing = report.missing_columns
```

### Pattern 2: Deterministic Upgrade Outcome Signaling

**What:** Explicitly distinguish `upgraded` vs `already_compatible` outcomes.
**When to use:** Script/runtime initialization flows where operators need clear migration state.
**Example:**

```python
result = upgrade_schema_compatibility(engine=engine)
if result.actions:
    status = "upgraded"
else:
    status = "already_compatible"
```

### Pattern 3: Readiness Degradation with Actionable Detail

**What:** Return `degraded`/`unavailable` service statuses with explicit missing columns.
**When to use:** `/health/detailed`, `/analytics/status`, and metrics readiness signals.
**Example:**

```python
ServiceStatus(
    name="schema_compatibility",
    status="unavailable",
    detail="missing: prediction_outcomes(predicted_price)",
)
```

### Anti-Patterns to Avoid

- Health checks that only verify DB file presence.
- Broad `except Exception: database_ok = False` without schema detail.
- Readiness paths that mutate schema as a side effect.

## Don’t Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Ad hoc schema checks in multiple endpoints | Repeated inspector snippets | Reuse `schema_compat` audit functions | Prevents drift and inconsistent output |
| Opaque readiness booleans | Generic `database_ok` only | Structured schema compatibility fields | Operator diagnostics become actionable |

## Common Pitfalls

### Pitfall 1: Upgrade checks that accidentally mutate data during readiness probes
**What goes wrong:** Health endpoint triggers schema changes unexpectedly.
**How to avoid:** Use `audit_schema_compatibility` only in readiness surfaces.

### Pitfall 2: Idempotency asserted only once
**What goes wrong:** First upgrade passes, second run regresses due output/logic drift.
**How to avoid:** Test repeated `--upgrade` and runtime initialization cycles with preserved row assertions.

### Pitfall 3: Swallowing schema errors under generic DB unavailable status
**What goes wrong:** Operators can’t tell if DB is missing or schema is incompatible.
**How to avoid:** Include schema-specific service status and missing-column details.

## Code Examples

### Current readiness limitation

```python
# src/bitbat/api/routes/health.py
if db_path.exists():
    return ServiceStatus(name="database", status="ok")
```

### Existing compatibility primitive to leverage

```python
# src/bitbat/autonomous/schema_compat.py
report = audit_schema_compatibility(engine=connection)
```

### Existing status fallback that hides root cause

```python
# src/bitbat/api/routes/analytics.py
except Exception:
    database_ok = False
```

## Open Questions

1. Should schema compatibility be a dedicated service in `/health/detailed` or folded into `database` service detail?
   - Recommendation: dedicated service for clearer diagnostics and alerting.
2. Should metrics endpoint expose schema compatibility gauge in this phase?
   - Recommendation: include if low effort; otherwise prioritize health/status endpoint correctness first.

## Sources

### Primary (HIGH confidence)
- `src/bitbat/autonomous/schema_compat.py`
- `src/bitbat/autonomous/db.py`
- `scripts/init_autonomous_db.py`
- `src/bitbat/api/routes/health.py`
- `src/bitbat/api/routes/analytics.py`
- `tests/autonomous/test_schema_compat.py`

### Secondary (MEDIUM confidence)
- `.planning/ROADMAP.md`
- `.planning/REQUIREMENTS.md`
- `.planning/STATE.md`
- `.planning/phases/01-schema-contract-baseline/01-VERIFICATION.md`

## Metadata

**Confidence breakdown:**
- Migration hardening approach: HIGH
- Readiness surface strategy: HIGH
- Detailed scope split between phase plans: MEDIUM-HIGH

**Research date:** 2026-02-24
**Valid until:** 2026-03-24
