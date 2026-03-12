# Phase 4: Monitor Flow Consistency & API Alignment - Research

**Researched:** 2026-02-24
**Domain:** Prediction semantic consistency across monitor writes, validation outcomes, API responses, and GUI timeline consumers
**Confidence:** HIGH

## Summary

Phase 4 should establish one canonical prediction semantic contract across monitor persistence, realization, and read surfaces. The current codebase has material semantic drift:

1. Monitor prediction writes persist `predicted_direction`, `predicted_return`, and `predicted_price`, but do not provide probability fields, while the DB layer defaults `p_up`/`p_down` to `0.0`. This fabricates "0% confidence" semantics instead of representing "confidence unavailable."
2. Validator computes correctness using sign(`predicted_return`) when available, but persistence (`AutonomousDB.realize_prediction`) recomputes correctness by direction equality only. In-memory validation summaries and stored DB outcomes can diverge.
3. API prediction responses omit probability/confidence fields consumed by timeline-oriented UX, while GUI timeline/widgets rely on DB `p_up`/`p_down` semantics and the Streamlit home page expects a `confidence` key not provided by widget data helpers.

These are direct blockers for `MON-02` and `API-01` because freq/horizon records can carry inconsistent semantics depending on whether consumers read validator output, persisted rows, API models, or GUI helpers.

**Primary recommendation:** implement a shared prediction semantic normalization path at monitor DB boundaries, make correctness persistence deterministic with validator logic, then align API schemas/read adapters and GUI consumers around the same confidence/direction contract.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| MON-02 | Prediction write/read paths in monitor + validator behave consistently across active freq/horizon settings | Normalize prediction persistence semantics (`p_up/p_down/p_flat`, `correct`) in shared DB boundary and enforce parity in validator + read paths |
| API-01 | API and GUI timeline-consumed fields remain semantically aligned | Extend read-model mapping (API + GUI helpers/timeline) to expose the same confidence/direction/outcome semantics without fabricated defaults |
</phase_requirements>

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| SQLAlchemy | 2.x | Prediction row persistence and realization update semantics | Existing autonomous DB boundary already centralizes monitor writes/reads |
| FastAPI + Pydantic | current repo baseline | API response model contract for prediction fields | Existing `/predictions` endpoints already map DB rows through `PredictionResponse` |
| pandas / sqlite3 | current repo baseline | GUI timeline and widget read-model shaping | Existing Streamlit timeline/widgets already rely on these read paths |
| pytest | current repo baseline | Regression protection for semantic consistency | Existing test suites across `tests/autonomous`, `tests/api`, `tests/gui` |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|-----------|-----------|----------|
| Canonical DB-boundary semantic normalization | Per-surface ad hoc mapping (API and GUI each infer semantics) | Faster short-term edits, but semantic drift reappears and MON-02/API-01 regressions become likely |
| Optional confidence (nullable when unavailable) | Always synthesize confidence from direction (`1.0` or `0.0`) | Easier UI display, but semantically misleading for regression outputs and degrades trust |

## Architecture Patterns

### Pattern 1: Canonical Prediction Semantics at Persistence Boundary

**What:** `store_prediction` should normalize direction/probability payload semantics once (including explicit "unavailable" behavior) before row write.
**When to use:** Every monitor prediction insert regardless of freq/horizon.

### Pattern 2: Single Correctness Derivation Path for Realization

**What:** Realized `correct` should derive from one rule shared by validator and DB persistence.
**When to use:** Validator batch updates and any future realization writes.

### Pattern 3: Read-Model Adapters for API/UI Contract Stability

**What:** API and GUI helpers should consume normalized persisted semantics and expose consistent optional confidence fields.
**When to use:** `/predictions` response mapping and timeline/widget read helpers.

## Code Evidence (Current State)

### Monitor write currently omits probability fields; DB defaults to zeros

```python
# src/bitbat/autonomous/predictor.py
self.db.store_prediction(..., predicted_return=predicted_return, predicted_price=predicted_price)

# src/bitbat/autonomous/db.py
def store_prediction(..., p_up: float = 0.0, p_down: float = 0.0, ...):
```

### Validator computes one correctness value, DB persistence computes another

```python
# src/bitbat/autonomous/validator.py
if prediction.predicted_return is not None:
    correct = pred_sign == actual_sign

# src/bitbat/autonomous/db.py
prediction.correct = prediction.predicted_direction == actual_direction
```

### API omits confidence/probability while GUI timeline relies on p_up/p_down

```python
# src/bitbat/api/schemas.py
class PredictionResponse(BaseModel):
    predicted_direction: str
    predicted_return: float | None = None
    predicted_price: float | None = None

# src/bitbat/gui/timeline.py
confidence = max(float(row.get("p_up", 0)), float(row.get("p_down", 0)))
```

## Pitfalls

### Pitfall 1: Fabricated confidence semantics from numeric defaults
If `p_up/p_down` defaults are numeric zeros rather than nullable semantics, timeline confidence and API-derived confidence can mislead operators.

### Pitfall 2: Mixed correctness logic causing cross-surface drift
If validator summary and persisted `correct` diverge, API performance metrics and GUI accuracy displays become inconsistent.

### Pitfall 3: Contract updates that break existing consumers
Adding confidence/probability fields without backward-compatible optionality can break current API clients/tests. Migration should be additive and nullable.

## Recommended Plan Split

1. **04-01 (Wave 1):** Monitor/validator semantic normalization at DB boundary (`MON-02` foundation).
2. **04-02 (Wave 2):** API + GUI read alignment on normalized semantics (`API-01`, dependent on 04-01 contract).

## Sources

### Primary (HIGH confidence)
- `src/bitbat/autonomous/predictor.py`
- `src/bitbat/autonomous/validator.py`
- `src/bitbat/autonomous/db.py`
- `src/bitbat/autonomous/models.py`
- `src/bitbat/api/routes/predictions.py`
- `src/bitbat/api/schemas.py`
- `src/bitbat/gui/timeline.py`
- `src/bitbat/gui/widgets.py`
- `streamlit/app.py`
- `tests/autonomous/test_validator.py`
- `tests/api/test_predictions.py`
- `tests/gui/test_timeline.py`
- `tests/gui/test_widgets.py`

### Secondary (MEDIUM confidence)
- `.planning/ROADMAP.md`
- `.planning/REQUIREMENTS.md`
- `.planning/STATE.md`

## Metadata

**Confidence breakdown:**
- Monitor write/read semantic drift diagnosis: HIGH
- Correctness derivation inconsistency diagnosis: HIGH
- API/GUI alignment and contract-update scope: HIGH

**Research date:** 2026-02-24
**Valid until:** 2026-03-24
