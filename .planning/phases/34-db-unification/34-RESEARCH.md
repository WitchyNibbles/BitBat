# Phase 34: DB Unification - Research

**Researched:** 2026-03-12
**Domain:** Autonomous DB access layer unification
**Confidence:** HIGH

## Summary

Phase 34 should standardize on the existing SQLAlchemy-based stack that already powers `bitbat.autonomous.db.AutonomousDB`, `bitbat.autonomous.models`, and the schema-compatibility helpers. The competing approach is not another ORM layer; it is a small set of runtime surfaces that still bypass the service layer with direct `sqlite3` queries. Those remaining raw call sites are concentrated in three files:

- `src/bitbat/api/routes/system.py`
- `src/bitbat/gui/widgets.py`
- `src/bitbat/gui/timeline.py`

The rest of the autonomous, CLI, API metrics/predictions, and retraining stack already uses `AutonomousDB` sessions and SQLAlchemy models. That means the phase is a migration, not a ground-up redesign: extend the existing DB service layer to support the read patterns currently implemented with raw SQL, then migrate the remaining raw-`sqlite3` consumers onto that layer.

There is a second, important issue beyond raw read access: several write flows still span multiple DB sessions. `AutoRetrainer` and `ContinuousTrainer` create retraining events, store model versions, activate models, and complete/fail events in separate transactions. That violates the Phase 34 context decisions around “all or nothing” semantics for retraining/model activation and event recording. The plans therefore need to cover both:

1. **read-path unification** — eliminate raw `sqlite3` runtime usage in `src/`
2. **transaction boundary unification** — consolidate write-side units of work that currently span multiple sessions

**Primary recommendation:** keep SQLAlchemy + `AutonomousDB` as the single database access layer. Add compatibility-aware read/query methods to `AutonomousDB` for the API system routes and Streamlit timeline/widgets, add a small transient-lock retry/circuit-breaker policy inside the DB layer, and add explicit atomic “unit of work” helpers for retraining/model promotion flows. Remove all runtime `sqlite3` imports from `src/bitbat/`.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| DEBT-03 | Dual DB access unified — SQLAlchemy ORM + raw sqlite3 consolidated into a single consistent approach | Existing SQLAlchemy layer already covers most runtime DB usage; only 3 runtime files still use raw `sqlite3`; retraining/event writes still need transaction consolidation |
</phase_requirements>

## Standard Stack

### Core (already installed — no new dependencies)

| Library | Purpose | Why it should remain the standard |
|---------|---------|-----------------------------------|
| `sqlalchemy` | Engine/session lifecycle, ORM models, `text()` / inspection helpers | Already used by `AutonomousDB`, schema compatibility, and runtime DB initialization |
| `bitbat.autonomous.db.AutonomousDB` | Runtime repository/service layer | Already used by monitor, retrainer, CLI monitor, API metrics/predictions |
| `bitbat.autonomous.models` | Declarative schema + engine creation | Already defines the authoritative schema and engine configuration |
| `bitbat.autonomous.schema_compat` | Legacy schema audit/upgrade behavior | Already preserves backward compatibility for existing `autonomous.db` files |

### Existing raw approach to retire

| Approach | Current usage | Recommendation |
|----------|---------------|----------------|
| `sqlite3.connect(...)` | `api/routes/system.py`, `gui/widgets.py`, `gui/timeline.py` | Remove from `src/`; route all runtime access through SQLAlchemy/`AutonomousDB` |

## Architecture Patterns

### Pattern 1: Single runtime DB layer = `AutonomousDB`

The codebase already has the right anchor:

- `AutonomousDB` owns engine creation, schema compatibility checks, and session lifecycle
- monitor/validator/predictor/drift/retrainer/CLI/API already instantiate it directly
- the phase should **extend** that layer instead of introducing a second repository abstraction

That means the migration target is:

- **all runtime database reads and writes in `src/bitbat/` go through SQLAlchemy/`AutonomousDB`**
- **zero runtime `sqlite3` imports or `sqlite3.connect()` calls remain in `src/bitbat/`**

### Pattern 2: Compatibility-aware read methods belong in `AutonomousDB`

The raw sqlite files are doing two things:

1. direct DB access
2. compatibility fallbacks for legacy columns (`created_at` vs `timestamp`, `prediction_timestamp` vs `timestamp_utc`, etc.)

Those compatibility rules should move into `AutonomousDB` as explicit read methods. The service layer can still use SQLAlchemy inspection and `text()` queries when an ORM-only query would be awkward, but those details stay inside the single DB layer.

Expected method families:

- system/API reads
  - list system logs
  - list retraining events
  - list performance snapshots
- widget/timeline reads
  - latest prediction summary
  - recent events
  - pair-filter options
  - timeline rows with legacy timestamp/probability fallbacks
  - system activity/status summary

**Key point:** using `sqlalchemy.inspect()` or `engine.connect()` inside `AutonomousDB` still counts as one DB approach. The anti-goal is raw `sqlite3` usage dispersed across runtime modules.

### Pattern 3: API fail-fast, GUI shape-preserving

The user locked fail-fast behavior for read-only endpoints. That applies cleanly to FastAPI routes:

- `/system/logs`
- `/system/retraining-events`
- `/system/performance-snapshots`

These routes should raise request errors when DB access fails rather than silently returning empty payloads.

The Streamlit helper functions have a different established contract, enforced by tests:

- `get_system_status(...)` returns a structured status dict
- `get_latest_prediction(...)` returns `None` on missing/unsupported DB
- `get_recent_events(...)` returns `[]`
- `get_timeline_data(...)` returns an empty DataFrame

Those empty-state contracts should stay stable even after the backend access path is unified. The unification is internal; the GUI helper surface should remain backward-compatible unless a behavior change is explicitly tested and intended.

### Pattern 4: Small atomic transactions for retraining/promotion

Current retraining flows span separate sessions:

- `ContinuousTrainer.retrain()` creates the event in one session, stores/activates models in another, then completes/fails the event in yet another
- `AutoRetrainer.retrain()` follows the same pattern

This should be refactored into explicit “unit of work” helpers inside `AutonomousDB`, for example:

- create/complete event + activate model in one transaction
- create/fail event updates in one transaction for the failure path

The user’s direction is **smaller atomic transactions per high-level phase**, not one giant transaction across the full command. That means:

- CLI subprocess work (`features build`, `model cv`, `model train`) remains outside the DB transaction
- DB state transitions for promotion/event bookkeeping should be atomic within a single session

### Pattern 5: Retry + circuit breaker live in the DB layer

The retry/circuit-breaker behavior for transient lock contention should not be reimplemented in API/CLI/monitor callers. The clean place is inside the unified DB layer so all consumers get the same policy and message shape.

Recommended shape:

- detect transient lock errors (for SQLite: “database is locked” / operational lock failures)
- retry a small fixed number of times with short backoff
- if retries are exhausted, open a short-lived circuit and surface an explicit “temporarily unavailable” / “circuit open” message
- keep the default user-facing surface short: one message + one hint line

## Current Runtime Inventory

### Raw `sqlite3` call sites still in `src/`

`rg -n "import sqlite3|sqlite3\\.connect|sqlite3\\.Connection" src/bitbat`

Results:

- `src/bitbat/api/routes/system.py`
- `src/bitbat/gui/widgets.py`
- `src/bitbat/gui/timeline.py`

### Existing SQLAlchemy service-layer usage already in place

Already using `AutonomousDB`:

- `src/bitbat/autonomous/agent.py`
- `src/bitbat/autonomous/predictor.py`
- `src/bitbat/autonomous/validator.py`
- `src/bitbat/autonomous/drift.py`
- `src/bitbat/autonomous/retrainer.py`
- `src/bitbat/autonomous/continuous_trainer.py`
- `src/bitbat/api/routes/predictions.py`
- `src/bitbat/api/routes/metrics.py`
- `src/bitbat/api/routes/analytics.py`
- `src/bitbat/cli/commands/monitor.py`
- `src/bitbat/cli/commands/validate.py`
- `src/bitbat/cli/commands/batch.py`

### Test coverage gap worth addressing

There are strong tests for:

- `AutonomousDB`
- GUI widgets/timeline helpers
- monitor/API metrics/predictions flows

There are **no direct API tests** for:

- `/system/logs`
- `/system/retraining-events`
- `/system/performance-snapshots`

Phase 34 should add those so the API migration is guarded before the raw sqlite path is removed.

## Don’t Hand-Roll

| Problem | Don’t Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| New ad-hoc DB abstraction | Separate “db_utils.py” wrapper around sqlite3 | Extend `AutonomousDB` | One existing runtime DB layer already exists |
| Raw SQL in routes/widgets | `sqlite3.connect()` in API/GUI modules | `AutonomousDB` query helpers | Keeps DB lifecycle and compatibility logic centralized |
| Per-caller retry logic | Retry loops in routes/commands | Unified DB-layer retry/circuit-breaker | Consistent behavior and messaging |
| Best-effort event/model promotion writes | Multiple independent sessions for one logical promotion | Single DB-layer transaction helper | Matches all-or-nothing context decision |

## Common Pitfalls

### Pitfall 1: Treating “no raw sqlite3 imports” as the whole phase
**What goes wrong:** The three raw read surfaces are migrated, but retraining/event writes still span multiple DB sessions and preserve inconsistent transaction behavior.  
**Why it matters:** Phase 34 also requires one connection lifecycle and one consistent query/transaction pattern.  
**How to avoid:** Include explicit plan work for write-side transaction helpers, not just read-side migrations.

### Pitfall 2: Breaking GUI empty-state contracts while making API fail fast
**What goes wrong:** A unified DB helper raises in every case, breaking timeline/widget functions that intentionally return `None`, `[]`, or empty DataFrames for missing local DB state.  
**How to avoid:** Separate fail-fast API behavior from GUI helper return-shape preservation. Same backend layer, different caller-facing contracts.

### Pitfall 3: Re-adding raw SQL compatibility logic outside the DB layer
**What goes wrong:** Routes/widgets stop importing `sqlite3`, but they grow their own column fallback logic or schema probing.  
**How to avoid:** Move compatibility-aware query logic into `AutonomousDB` and keep callers thin.

### Pitfall 4: One huge transaction around subprocess-based retraining
**What goes wrong:** The planner tries to put external training commands and DB updates inside a single database transaction.  
**Why it fails:** The subprocess work is long-running and not transaction-safe.  
**How to avoid:** Use small atomic transactions for DB state transitions only.

### Pitfall 5: Missing structural enforcement
**What goes wrong:** One `sqlite3.connect()` survives in `src/`, and the phase is declared complete anyway.  
**How to avoid:** Add a structural regression test that fails if runtime `src/bitbat/` still contains raw `sqlite3` access.

## Code Examples

### Current raw API route pattern to retire

```python
con = _get_connection()
rows = con.execute(sql, params).fetchall()
con.close()
```

This lives in `src/bitbat/api/routes/system.py` and should be replaced by `AutonomousDB` read methods plus standardized request-level failure handling.

### Current raw GUI helper pattern to retire

```python
con = sqlite3.connect(str(db_path))
rows = con.execute(sql, params).fetchall()
con.close()
```

This pattern appears in `src/bitbat/gui/widgets.py` and `src/bitbat/gui/timeline.py`. The query logic can stay compatibility-aware, but the runtime access path should move behind the unified DB layer.

### Current write-side session split to fix

`AutoRetrainer.retrain()` currently:

1. creates retraining event in one session
2. stores model version in another session
3. activates model in another session
4. completes/fails event in another session

`ContinuousTrainer.retrain()` has the same shape.

These should become explicit DB-layer unit-of-work helpers so model activation and event completion are atomic together.

## Validation Architecture

### Test Framework

Use the existing `pytest` suites already covering:

- DB layer: `tests/autonomous/test_db.py`, `tests/autonomous/test_schema_compat.py`
- API layer: add dedicated `tests/api/test_system.py`
- GUI layer: `tests/gui/test_widgets.py`, `tests/gui/test_timeline.py`, `tests/gui/test_complete_gui.py`
- monitor/CLI regression: `tests/autonomous/test_agent_integration.py`, `tests/test_cli.py`

### Required Validation Gates

1. **Structural gate**
   - no runtime `sqlite3` imports / `sqlite3.connect` calls remain in `src/bitbat/`

2. **Schema compatibility gate**
   - existing legacy `autonomous.db` fixtures still load without migration

3. **API behavior gate**
   - `/system/*` read endpoints pass through unified layer and fail requests on DB errors

4. **GUI behavior gate**
   - timeline/widgets preserve empty-return contracts and legacy-column compatibility

5. **Transaction gate**
   - retraining/model activation + event completion are atomic inside one DB transaction

### Recommended Verification Commands

- `poetry run pytest tests/autonomous/test_db.py tests/autonomous/test_schema_compat.py -x`
- `poetry run pytest tests/api/test_system.py tests/api/test_metrics.py tests/api/test_predictions.py -x`
- `poetry run pytest tests/gui/test_widgets.py tests/gui/test_timeline.py tests/gui/test_complete_gui.py -x`
- `poetry run pytest tests/autonomous/test_agent_integration.py tests/autonomous/test_retrainer.py tests/test_cli.py -x`

## Open Questions

None blocking for planning. The user has already locked the important product-level decisions:

- fail-fast API request behavior
- short standardized diagnostics with hint line
- explicit circuit-breaker messaging
- removal of raw sqlite fallback paths
- existing DB files must keep working
- smaller atomic transactions with all-or-nothing DB state transitions
