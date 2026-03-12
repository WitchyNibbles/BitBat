# Phase 34: DB Unification - Context

**Gathered:** 2026-03-12
**Status:** Ready for planning

<domain>
## Phase Boundary

Unify all access to `autonomous.db` behind a single consistent database approach, removing the split between SQLAlchemy ORM and raw `sqlite3` usage. This phase preserves the existing schema and operator-facing surface area while standardizing query patterns, connection lifecycle, and transaction behavior.

</domain>

<decisions>
## Implementation Decisions

### Failure behavior
- Read-only DB-backed endpoints should fail the request instead of returning degraded payloads.
- Temporary DB lock issues should retry briefly using circuit-breaker-style behavior.
- Monitor startup should abort immediately if it detects a DB problem.
- Commands that combine filesystem work with DB work must pass DB preflight before any side effects occur.

### Compatibility posture
- Prioritize making the new unified path work cleanly rather than preserving mixed-access quirks.
- Existing `autonomous.db` files must continue to open and operate without schema migration.
- Low-level exception types may change if user-facing behavior stays coherent.
- Legacy raw `sqlite3` access paths should be removed, not kept as fallback behavior.
- Tests should be updated to reflect the unified behavior instead of preserving old split-access edge cases.

### Diagnostics style
- Default DB failure output should use one short message plus one hint line.
- Low-level exception names and internals should stay hidden unless debug/logging output exposes them.
- Circuit-breaker-open conditions should be stated explicitly to the operator.
- CLI, API, and monitor flows should share a standardized wording style for DB-related errors.

### Runtime consistency
- Monitor cycles should remain consistent and avoid partial DB state when writes fail.
- Retraining/model activation and event recording should be atomic together.
- Failed transactions should report failure without best-effort partial persistence.
- Prefer smaller atomic transactions per high-level phase of work rather than one large transaction for an entire command or run.

### Claude's Discretion
- Exact retry thresholds, breaker timing, and reset policy for temporary lock handling.
- The specific unified DB layer shape, as long as raw `sqlite3` call sites are removed.
- The exact standardized wording template, provided it stays short, clear, and consistent.

</decisions>

<specifics>
## Specific Ideas

- The unification should be strict: fail fast by default, not limp along in degraded mode.
- Operator messaging should stay clean and actionable rather than surfacing internal DB library details.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 34-db-unification*
*Context gathered: 2026-03-12*
