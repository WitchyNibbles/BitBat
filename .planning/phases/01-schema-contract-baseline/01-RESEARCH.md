# Phase 1: Schema Contract Baseline - Research

**Researched:** 2026-02-24
**Domain:** SQLite schema compatibility for autonomous prediction runtime
**Confidence:** HIGH

## Summary

The operational failure (`no such column: prediction_outcomes.predicted_price`) is a classic brownfield schema drift issue: runtime code and ORM models expect a column that older local SQLite files do not contain. The current initialization path (`Base.metadata.create_all`) creates missing tables but does not mutate existing table schemas, so historical DBs remain incompatible.

Phase 1 should deliver a compatibility baseline with two capabilities: (1) deterministic schema introspection and additive upgrade for required columns, and (2) startup preflight checks that block unsafe monitor runs with clear remediation output. This should be implemented without destructive resets so existing prediction history survives.

**Primary recommendation:** Introduce a dedicated `schema_compat` module for required-column contract enforcement and call it from initialization/preflight paths before monitor IO.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| SCHE-01 | Existing local DBs can be upgraded safely to include required runtime columns | Additive `ALTER TABLE ... ADD COLUMN` guarded by column existence checks |
| SCHE-02 | Startup validates schema compatibility and surfaces actionable errors | Preflight validation invoked before monitor loop and surfaced through CLI/runtime diagnostics |
</phase_requirements>

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| SQLAlchemy | 2.x | ORM metadata, engine/session lifecycle | Existing authoritative schema/runtime layer in repo |
| SQLite | bundled | Persistent local monitor state | Current default DB backend for monitor, API, and GUI surfaces |
| Python stdlib / SQL introspection | stdlib | Idempotent compatibility checks and upgrade guards | Minimal dependency path, deterministic for local DB files |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest | 8.x | Compatibility and preflight regression tests | Required to prevent re-introducing schema drift failures |
| Alembic (defer decision) | latest compatible | Long-term migration revision history | Consider in later hardening phase if migration complexity grows |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|-----------|-----------|----------|
| Additive compatibility layer | Drop/recreate table or DB | Faster short-term but destroys history and masks migration bugs |
| Immediate full Alembic adoption | Lightweight compatibility helper in existing module boundaries | Alembic is stronger long-term; helper unblocks Phase 1 with lower implementation overhead |

## Architecture Patterns

### Recommended Project Structure

```text
src/bitbat/autonomous/
├── models.py                 # ORM schema truth
├── db.py                     # repository operations and session handling
├── schema_compat.py          # NEW: required columns, introspection, additive upgrades
└── agent.py                  # monitor preflight usage

scripts/
└── init_autonomous_db.py     # explicit compatibility audit/upgrade entrypoint

tests/autonomous/
└── test_schema_compat.py     # compatibility + idempotency regression tests
```

### Pattern 1: Required-Column Registry

**What:** Maintain explicit required column set for runtime-critical tables.
**When to use:** Any runtime path depending on `prediction_outcomes` fields.
**Example:**

```python
REQUIRED_PREDICTION_COLUMNS = {
    "id", "timestamp_utc", "predicted_direction", "predicted_return",
    "predicted_price", "model_version", "freq", "horizon"
}
```

### Pattern 2: Additive Idempotent Upgrade

**What:** Add missing nullable columns only when absent.
**When to use:** Legacy DB detected at startup/audit.
**Example:**

```python
if "predicted_price" not in current_cols:
    conn.execute(text("ALTER TABLE prediction_outcomes ADD COLUMN predicted_price FLOAT"))
```

### Pattern 3: Startup Preflight Gate

**What:** Validate compatibility before monitor loop proceeds.
**When to use:** `monitor start`, `monitor run-once`, monitor-agent bootstrap.
**Example:**

```python
missing = required_cols - current_cols
if missing:
    raise RuntimeError(f"Schema incompatible. Missing columns: {sorted(missing)}")
```

### Anti-Patterns to Avoid

- Assuming `create_all` handles migration for existing tables.
- Applying destructive reset as default fix path.
- Catch-and-continue handling for schema incompatibility in critical runtime paths.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Compatibility checks scattered in many modules | Repeated ad-hoc query snippets | Single shared compatibility utility module | Consistent behavior + easier regression test coverage |
| Operator remediation logic embedded in stack traces | Opaque unstructured exceptions | Structured error messages with remediation hints | Faster incident resolution |

**Key insight:** A small centralized compatibility boundary in `autonomous/` is enough to restore runtime reliability for Phase 1.

## Common Pitfalls

### Pitfall 1: One-time fix without regression tests
**What goes wrong:** Column added manually once; issue returns on fresh/dev DB variants.
**How to avoid:** Automated tests for legacy schema + idempotent rerun behavior.

### Pitfall 2: Upgrade path that mutates data semantics
**What goes wrong:** Existing rows or constraints are unintentionally changed.
**How to avoid:** Additive nullable column additions only in Phase 1.

### Pitfall 3: Preflight implemented in one entrypoint only
**What goes wrong:** `monitor start` works but `monitor run-once` or script path still fails.
**How to avoid:** Reuse one preflight utility across all monitor entrypoints.

## Code Examples

### ORM expectation already present

```python
# src/bitbat/autonomous/models.py
predicted_price = mapped_column(Float, nullable=True)
```

### Current DB deficiency observed

```text
PRAGMA table_info(prediction_outcomes)
# missing: predicted_price
```

### Runtime read dependency

```python
# src/bitbat/gui/widgets.py
SELECT ... predicted_price ... FROM prediction_outcomes
```

## Open Questions

1. Should compatibility upgrade execute automatically in production monitor boot, or require explicit operator flag?
   - Recommendation: permit automatic additive-safe upgrade, then enforce validation.
2. Should health endpoint report compatibility state in Phase 1 or Phase 2?
   - Recommendation: keep detailed readiness projection in Phase 2 per roadmap traceability.

## Sources

### Primary (HIGH confidence)
- `src/bitbat/autonomous/models.py`
- `src/bitbat/autonomous/db.py`
- `src/bitbat/autonomous/agent.py`
- `scripts/init_autonomous_db.py`
- live DB introspection (`data/autonomous.db`, `PRAGMA table_info(prediction_outcomes)`)

### Secondary (MEDIUM confidence)
- `.planning/REQUIREMENTS.md`
- `.planning/ROADMAP.md`
- `.planning/STATE.md`

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH (derived directly from existing code)
- Architecture patterns: HIGH (aligned with current module boundaries)
- Pitfalls: HIGH (rooted in reproduced runtime mismatch)

**Research date:** 2026-02-24
**Valid until:** 2026-03-24
