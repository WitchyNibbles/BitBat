# Phase 1: Schema Contract Baseline - Research

**Researched:** 2026-02-24
**Domain:** SQLite schema compatibility for autonomous prediction runtime
**Confidence:** HIGH

## Summary

The immediate runtime failure is a schema/code mismatch: the `PredictionOutcome` ORM model and downstream reads/writes expect `prediction_outcomes.predicted_price`, but the existing `data/autonomous.db` table does not contain that column. Current initialization (`Base.metadata.create_all`) creates missing tables but does not migrate existing table schemas, so long-lived databases remain incompatible.

Phase 1 should establish a compatibility baseline that works on existing databases without destructive resets. The safest near-term approach is an idempotent compatibility upgrade path (column existence checks + additive migrations) plus startup preflight checks that fail fast with actionable error messages.

**Primary recommendation:** Add an explicit schema compatibility layer for `prediction_outcomes` (including `predicted_price`) and run it before monitor cycles start.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| SCHE-01 | Existing local DBs can be upgraded safely to include required runtime columns | Additive, idempotent migration strategy using SQLite introspection + `ALTER TABLE ... ADD COLUMN` guards |
| SCHE-02 | Startup validates schema compatibility and surfaces actionable errors | Preflight schema check in monitor startup path + clear diagnostics when mismatch persists |
</phase_requirements>

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| SQLAlchemy | 2.x | ORM model definitions and sessions | Already authoritative model layer in this codebase |
| SQLite | bundled | Runtime monitor state storage | Current persistent backend for local operations |
| Python `sqlite3`/PRAGMA via SQLAlchemy engine | stdlib | Column introspection and migration guards | Deterministic compatibility checks for existing DB files |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Alembic (optional in follow-up) | latest compatible | Structured migration revisions | Use if moving from compatibility patching to full migration history |
| pytest | 8.x | Regression verification | Use for compatibility + preflight behavior coverage |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|-----------|-----------|----------|
| Additive compatibility patching | Full DB reset/recreate | Reset is simpler but destroys history and hides migration regressions |
| Immediate Alembic adoption | Lightweight migration helper in current modules | Alembic gives long-term rigor; helper is faster for critical unblock |

## Architecture Patterns

### Recommended Project Structure

```text
src/bitbat/autonomous/
├── models.py                 # ORM schema truth
├── db.py                     # runtime repository operations
├── schema_compat.py          # NEW: compatibility checks + idempotent upgrades
└── agent.py                  # monitoring startup path calls preflight

scripts/
└── init_autonomous_db.py     # can invoke schema compatibility command

tests/autonomous/
└── test_schema_compat.py     # NEW: regression tests for DB upgrade paths
```

### Pattern 1: Additive Migration Guard

**What:** Introspect table columns, add missing non-destructive columns only when absent.
**When to use:** Existing SQLite DB might be behind current ORM model.
**Example:**

```python
# Pseudocode pattern
columns = existing_columns("prediction_outcomes")
if "predicted_price" not in columns:
    execute("ALTER TABLE prediction_outcomes ADD COLUMN predicted_price FLOAT")
```

### Pattern 2: Startup Schema Preflight

**What:** Validate required columns before monitor runtime operations.
**When to use:** Before monitor loop and DB-dependent CLI/API paths.
**Example:**

```python
missing = required_columns - existing_columns("prediction_outcomes")
if missing:
    raise RuntimeError(f"Schema incompatible: missing {sorted(missing)}")
```

### Anti-Patterns to Avoid

- Running monitor writes before schema preflight.
- Using exception swallowing to hide compatibility failures.
- Treating `create_all` as a migration tool for existing tables.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Schema drift detection | Scattered ad-hoc checks in many call sites | Single schema compatibility module + shared utility | Centralized behavior, easier regression testing |
| Startup failure messaging | Generic stack traces only | Structured actionable errors | Operators can fix quickly without code spelunking |

**Key insight:** Centralized compatibility logic prevents repeated break/fix cycles across monitor, API, and GUI.

## Common Pitfalls

### Pitfall 1: Assuming `create_all` upgrades existing schemas
**What goes wrong:** Existing DB tables keep old columns; runtime queries fail.
**How to avoid:** Explicit migration/introspection for table alterations.

### Pitfall 2: Fixing only one codepath
**What goes wrong:** Monitor works, but UI/API still fail due unpatched read paths.
**How to avoid:** Define required-column contract once and reuse across surfaces.

### Pitfall 3: Non-idempotent migration logic
**What goes wrong:** Re-running startup fails or mutates schema unexpectedly.
**How to avoid:** Guard every migration with presence checks.

## Code Examples

### Existing schema truth source

```python
# src/bitbat/autonomous/models.py
class PredictionOutcome(Base):
    predicted_price = mapped_column(Float, nullable=True)
```

### Existing runtime expectation

```python
# src/bitbat/gui/widgets.py query includes predicted_price
SELECT timestamp_utc, predicted_direction, predicted_return, predicted_price, ...
```

### Observed live DB mismatch

```text
prediction_outcomes columns in current data/autonomous.db:
id, timestamp_utc, prediction_timestamp, predicted_direction, p_up,
p_down, p_flat, predicted_return, actual_return, actual_direction,
correct, model_version, freq, horizon, features_used, created_at, realized_at
# predicted_price missing
```

## Open Questions

1. Should compatibility upgrades run automatically at monitor startup or only via explicit command?
   - Recommendation: Startup preflight can apply additive-safe upgrades, then verify.
2. Should Alembic be introduced in Phase 1 or deferred to later hardening?
   - Recommendation: Keep Phase 1 focused on additive unblock; evaluate Alembic in later phase.

## Sources

### Primary (HIGH confidence)
- `src/bitbat/autonomous/models.py`
- `src/bitbat/autonomous/db.py`
- `src/bitbat/gui/widgets.py`
- `scripts/init_autonomous_db.py`
- live schema inspection of `data/autonomous.db` (`PRAGMA table_info(prediction_outcomes)`)

### Secondary (MEDIUM confidence)
- `.planning/REQUIREMENTS.md`
- `.planning/ROADMAP.md`

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH (directly from current codebase)
- Architecture: HIGH (fits existing module boundaries)
- Pitfalls: HIGH (reproduced by observed runtime mismatch)

**Research date:** 2026-02-24
**Valid until:** 2026-03-24
