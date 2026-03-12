# Phase 3: Monitor Runtime Error Elimination - Research

**Researched:** 2026-02-24
**Domain:** Autonomous monitor runtime DB fault elimination and critical-path error surfacing
**Confidence:** HIGH

## Summary

Phase 1 and Phase 2 established schema compatibility contracts and readiness checks, but runtime monitor flow still contains broad exception boundaries that can hide DB root causes during prediction and validation steps. The most important remaining risk is monitor cycles continuing after critical DB faults with only generic logs, making failure diagnosis slow and inconsistent.

Phase 3 should formalize DB fault handling in monitor-critical paths (`predictor`, `validator`, `agent`) so SQLAlchemy/runtime DB failures are captured with step-level context and surfaced to operators with actionable remediation guidance. This directly addresses MON-01 by eliminating recurring runtime `OperationalError` paths caused by schema drift, and MON-03 by ensuring critical failures are no longer silently swallowed.

**Primary recommendation:** add a shared monitor DB fault classification path, refactor `MonitoringAgent.run_once` exception boundaries to surface critical failures deterministically, and add regression tests that assert actionable diagnostics (step, error class, remediation hint) for DB failure scenarios.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| MON-01 | Monitoring cycles run without `OperationalError` caused by missing prediction columns | Add explicit schema/missing-column fault classification in predictor/validator DB interactions and regression coverage for legacy-schema failure paths |
| MON-03 | Critical monitor DB failures are surfaced with operator-actionable diagnostics | Promote critical DB faults through structured monitor result/log payloads and CLI/script-visible error messages |
</phase_requirements>

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| SQLAlchemy | 2.x | Runtime DB sessions and exception types (`OperationalError`) | Existing autonomous DB abstraction layer |
| Python logging | stdlib | Step-level runtime diagnostics | Already used by monitor, predictor, validator, and agent scripts |
| pytest | current repo baseline | Regression tests for runtime fault paths | Existing test pattern in `tests/autonomous/` and `tests/test_cli.py` |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `schema_compat` utilities | local | Missing-column diagnostics and remediation hints | Any DB failure where schema drift is suspected |
| `SystemLog` table (`AutonomousDB.log`) | local | Structured operator-visible diagnostic events | Persisting monitor failure context for GUI/API visibility |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|-----------|-----------|----------|
| Step-specific DB fault classification | Generic `except Exception` logging only | Simpler code, but loses remediation context and encourages silent degradation |
| Fail-fast on critical DB faults | Continue monitoring cycle after DB exceptions | More resilient to non-critical issues, but unsafe for data-integrity-critical DB write/read paths |

## Architecture Patterns

### Recommended Project Structure

```text
src/bitbat/autonomous/
├── db.py                      # DB operation boundary + structured log helper
├── predictor.py               # Prediction read/write DB fault classification
├── validator.py               # Validation write-path DB fault classification
├── agent.py                   # Critical-path exception boundaries + surfaced result
└── schema_compat.py           # Missing-column diagnostics and remediation text

scripts/
└── run_monitoring_agent.py    # Top-level surfacing of critical monitor DB failures

tests/
├── autonomous/test_agent_integration.py
├── autonomous/test_validator.py
└── test_cli.py
```

### Pattern 1: Critical DB Fault Classification at Source

**What:** Convert raw SQLAlchemy runtime DB exceptions into monitor-domain errors with operation metadata.
**When to use:** DB read/write operations in predictor and validator critical steps.
**Example:**

```python
try:
    self.db.store_prediction(...)
except OperationalError as exc:
    raise MonitorDatabaseError(step="predict_store", remediation="run --audit/--upgrade") from exc
```

### Pattern 2: Explicit Critical-Boundary Semantics in Monitor Agent

**What:** Separate recoverable data-source issues from critical DB persistence/validation failures.
**When to use:** `MonitoringAgent.run_once` orchestration boundary.
**Example:**

```python
except MonitorDatabaseError as exc:
    logger.exception("Critical monitor DB failure at %s", exc.step)
    raise
```

### Pattern 3: Operator-Actionable Diagnostics

**What:** Surface failure details with step name, exception class, missing columns/remediation, and target DB URL.
**When to use:** CLI monitor commands and `run_monitoring_agent.py` top-level loop.
**Example:**

```python
{
  "step": "predict_store",
  "exception": "OperationalError",
  "detail": "prediction_outcomes(predicted_price) missing",
  "remediation": "python scripts/init_autonomous_db.py --upgrade",
}
```

### Anti-Patterns to Avoid

- Catching broad exceptions in monitor critical paths and only logging generic messages.
- Returning `None` for critical DB persistence failures that should stop or explicitly flag cycle failure.
- Surfacing schema compatibility hints only at startup while runtime paths hide DB faults.

## Don’t Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Ad hoc SQL error parsing in multiple modules | Repeated regex/string checks in each file | Shared DB fault classification utility in autonomous layer | Keeps behavior consistent and testable |
| Unstructured diagnostic blobs | Free-form log strings only | Structured diagnostic payloads (`step`, `detail`, `remediation`) | Enables deterministic assertions and operator playbooks |

## Common Pitfalls

### Pitfall 1: Exception handling that masks critical step identity
**What goes wrong:** Logs show "prediction generation failed" but not whether failure occurred on query, insert, or commit.
**How to avoid:** Attach explicit monitor step IDs to raised/propagated errors.

### Pitfall 2: Runtime schema drift treated as generic DB outage
**What goes wrong:** Operator cannot distinguish missing-column schema issue from transient DB lock/IO issue.
**How to avoid:** Reuse schema compatibility detail formatting for missing-column failures.

### Pitfall 3: Tests covering startup compatibility but not runtime fault boundaries
**What goes wrong:** Startup preflight passes, but runtime write/read path regressions reintroduce silent failures.
**How to avoid:** Add focused runtime regression tests for predictor/validator/agent DB fault propagation.

## Code Examples

### Broad catch in critical prediction boundary (current)

```python
# src/bitbat/autonomous/agent.py
except Exception:
    logger.exception("Prediction generation failed")
```

### Top-level loop swallows root cause detail into generic heartbeat error (current)

```python
# scripts/run_monitoring_agent.py
except Exception:
    logger.exception("Monitoring cycle failed.")
```

### Existing schema diagnostics utility to reuse (current)

```python
# src/bitbat/autonomous/schema_compat.py
format_missing_columns(report)
```

## Open Questions

1. Should critical DB faults terminate `run_forever` loop or continue with explicit degraded status?
   - Recommendation: continue loop but always emit explicit structured error and high-severity alert; ensure failure is visible and not silent.
2. Should diagnostics be persisted only to logs or also to `system_logs` DB table?
   - Recommendation: both; console/file logs for operators plus DB persistence for API/UI observability.

## Sources

### Primary (HIGH confidence)
- `src/bitbat/autonomous/agent.py`
- `src/bitbat/autonomous/predictor.py`
- `src/bitbat/autonomous/validator.py`
- `src/bitbat/autonomous/db.py`
- `src/bitbat/autonomous/schema_compat.py`
- `scripts/run_monitoring_agent.py`
- `tests/autonomous/test_agent_integration.py`
- `tests/autonomous/test_validator.py`
- `tests/test_cli.py`

### Secondary (MEDIUM confidence)
- `.planning/ROADMAP.md`
- `.planning/REQUIREMENTS.md`
- `.planning/STATE.md`

## Metadata

**Confidence breakdown:**
- Runtime DB failure modes and boundary analysis: HIGH
- Required plan split across 03-01/02/03: HIGH
- Operator diagnostics design detail: MEDIUM-HIGH

**Research date:** 2026-02-24
**Valid until:** 2026-03-24
