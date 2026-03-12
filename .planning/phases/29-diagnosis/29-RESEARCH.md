# Phase 29: Diagnosis - Research

**Researched:** 2026-03-08
**Domain:** ML pipeline root-cause analysis — accuracy collapse in live prediction system
**Confidence:** HIGH (primary findings from direct code inspection and live database queries)

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| DIAG-01 | Operator can identify which pipeline stage caused the live accuracy collapse (ingestion → features → labels → model → serving) | Root cause fully localized to model+serving stages. CLI commands and test scripts identified to surface each boundary. |
| DIAG-02 | Root cause is documented with a reproducible trace (CLI run or test) before any fix is applied | Three distinct bugs identified. Each has a reproducible trace via CLI commands and database queries documented below. |
</phase_requirements>

---

## Summary

The live directional accuracy collapse to ~14% (38 correct / 266 realized) has been fully diagnosed through direct code inspection and live database analysis. Three compounding bugs are responsible, each at a distinct pipeline stage.

**Bug 1 — Model stage (PRIMARY):** `train.py` uses `objective: reg:squarederror` (regression) to train on `r_forward` (forward returns as float). This is a regression objective that predicts continuous return magnitude, not directional class probabilities. The model is trained on labels that include "up", "down", and "flat" classes but the actual training target passed is the raw float `r_forward` value — making the model optimize for return magnitude prediction, not directional correctness.

**Bug 2 — Serving/inference stage:** `infer.py::predict_bar` derives `predicted_direction` from `sign(predicted_return)` with no `tau` threshold. The model output (regression output) is biased negative (mean = -0.00246, 75% of predictions negative). This causes 203/268 predictions (76%) to be labeled "down" regardless of actual market direction.

**Bug 3 — Validation/serving stage:** `validator.py::classify_direction` uses `self.tau = 0.0` (hardcoded, ignoring the `tau` param). This means the validator classifies any positive actual return as "up" and any negative actual return as "down", producing zero "flat" actual outcomes from non-zero moves. However, 179 out of 266 realized predictions show `actual_return = 0.0` exactly — meaning price data lookup is returning the exact same price for start and end timestamps. This corrupts the validation signal by producing artificial "flat" actual outcomes.

**Primary recommendation:** The diagnosis deliverable for this phase is a committed root-cause document (`ROOT_CAUSE.md`) with the three bugs named, their pipeline stage, and a reproducible CLI trace for each — produced before any fix code is merged.

---

## Standard Stack

This phase is an investigation and documentation phase, not an implementation phase. No new libraries are required. All tooling already exists in the project.

### Core (already present)
| Tool | Version | Purpose | Notes |
|------|---------|---------|-------|
| `sqlite3` (stdlib) | — | Direct autonomous.db query | Used for live diagnosis queries |
| `poetry run bitbat` | project | CLI commands for pipeline inspection | `validate run`, `monitor status`, `monitor run-once` |
| `pytest` | project | Automated reproduction tests | Existing test suite + new diagnostic test |
| `xgboost` | project | Model inspection (`load_model`, `save_config`) | Inspect objective stored in artifact |

### Supporting
| Tool | Purpose | When to Use |
|------|---------|-------------|
| `pandas` + `parquet` | Inspect dataset boundary outputs | Verify label distribution, feature values |
| `json` stdlib | Parse model config | Confirm `objective` field in xgb.json |

**No installation needed.** All tools are in the existing Poetry environment.

---

## Architecture Patterns

### Pipeline Boundary Inspection Pattern

The diagnosis must confirm each pipeline boundary produces correct data before moving to the next. Use this inspection order:

```
Stage 1: Ingestion
  data/raw/prices/  → verify OHLCV completeness, no gaps

Stage 2: Features
  data/features/{freq}_{horizon}/dataset.parquet → verify feature values, no leakage

Stage 3: Labels
  data/features/{freq}_{horizon}/dataset.parquet:label column → verify distribution
  data/features/{freq}_{horizon}/meta.json → verify label_mode, tau used

Stage 4: Model
  models/{freq}_{horizon}/xgb.json → verify objective, feature names
  Inspect: json.loads(booster.save_config())["learner"]["objective"]["name"]

Stage 5: Serving (inference)
  autonomous.db: prediction_outcomes → verify predicted_direction distribution
  Check: sign(predicted_return) mapping to direction without tau

Stage 6: Validation (realization)
  autonomous.db: prediction_outcomes → verify actual_return=0.0 count
  Check: validator.tau hardcoded to 0.0 ignoring config tau
```

### Root-Cause Document Pattern

The DIAG-02 requirement is a committed document, not just a comment. Follow this structure for `ROOT_CAUSE.md`:

```markdown
# Root Cause Analysis: Live Accuracy Collapse

**Date:** YYYY-MM-DD
**Phase:** 29 — Diagnosis
**Committed before any fix code**

## Observed Symptom
38 correct / 266 realized = 14.3% directional accuracy (random baseline: 33%)

## Pipeline Stage Trace

### Stage N: [Stage Name]
**Bug:** [description]
**File:** [src/bitbat/...]
**Line:** [line number]
**Reproducible with:** [CLI command or test]
**Evidence:** [query result or output]

## Summary Table
| Stage | Bug | Severity | Fix Phase |
|-------|-----|----------|-----------|
| Model | ... | PRIMARY | Phase 30 |
| Serving | ... | SECONDARY | Phase 30 |
| Validation | ... | TERTIARY | Phase 30 |
```

### Diagnostic CLI Commands Pattern

Each stage has a CLI command or script that can be run to confirm the bug:

```bash
# Stage 4 — Model objective
poetry run python -c "
import xgboost as xgb, json
b = xgb.Booster()
b.load_model('models/5m_30m/xgb.json')
cfg = json.loads(b.save_config())
print(cfg['learner']['objective']['name'])
"
# Expected if broken: reg:squarederror
# Expected if fixed: multi:softprob

# Stage 5 — Serving direction distribution
poetry run python -c "
import sqlite3
conn = sqlite3.connect('data/autonomous.db')
cur = conn.cursor()
cur.execute('SELECT predicted_direction, COUNT(*) FROM prediction_outcomes GROUP BY predicted_direction')
print(dict(cur.fetchall()))
"
# Expected if broken: {'down': 203, 'up': 65} — heavily biased to down

# Stage 6 — Validation actual_return=0 corruption
poetry run python -c "
import sqlite3
conn = sqlite3.connect('data/autonomous.db')
cur = conn.cursor()
cur.execute('SELECT COUNT(*) FROM prediction_outcomes WHERE actual_return = 0.0')
print('Zero-return realizations:', cur.fetchone()[0])
cur.execute('SELECT COUNT(*) FROM prediction_outcomes WHERE actual_direction=\"flat\"')
print('Flat outcomes:', cur.fetchone()[0])
"
# Expected if broken: ~179 zero-return realizations
```

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Inspecting XGBoost objective | Custom JSON parser | `xgb.Booster.save_config()` + `json.loads` | Official API, returns full learner config |
| Database queries for diagnosis | ORM layer | Direct `sqlite3` | Simpler, no schema init required for read-only |
| Label distribution check | Custom script | `pd.read_parquet(...).label.value_counts()` | Already works with existing data contract |

**Key insight:** This phase is about observation and documentation, not new tooling. Use the existing CLI and Python stdlib to surface the evidence.

---

## Common Pitfalls

### Pitfall 1: Confusing training target with training label
**What goes wrong:** The dataset has both `r_forward` (float return) and `label` (string: up/down/flat). The model is trained on `r_forward` as the `y` target — the label column is never used for XGBoost training. This is the primary bug.
**Why it happens:** `fit_xgb` accepts any `pd.Series` as `y_train`. When the CLI builds the dataset with `label_mode=return_direction`, it assembles `r_forward` as the numeric target for XGBoost. The labels are used only for evaluation display.
**How to avoid:** Document this clearly in ROOT_CAUSE.md. The fix (Phase 30) must change training to use class labels with `multi:softprob`.
**Warning signs:** `objective=reg:squarederror` in the saved model config; label column is "label" not numeric.

### Pitfall 2: Conflating the tau=0.0 bug with the zero-return corruption bug
**What goes wrong:** Two separate bugs both cause "flat" actual directions — (a) `self.tau = 0.0` in validator should map small returns to "flat" but instead ALL non-zero moves go up/down; (b) `actual_return = 0.0` for 179 rows means the price lookup returned the same price for start and end.
**Why it happens:** The validator fetches prices from `data/raw/prices/` using `get_price_at_timestamp` with a 60-minute tolerance window. When a precise timestamp is not found and the closest match has the same close price, `calculate_return(p, p) = 0.0` → `classify_direction(0.0, tau=0.0) = "flat"`.
**How to avoid:** Document both bugs separately in ROOT_CAUSE.md with distinct traces. Check `prices_root.glob("**/*_{freq}.parquet")` data availability for the prediction window.
**Warning signs:** `actual_return = 0.0` exactly in 179/266 rows is a strong indicator of price lookup failure, not market behavior.

### Pitfall 3: Diagnosing without checking both freq/horizon pairs
**What goes wrong:** The pipeline runs for both `5m/30m` (188 predictions) and `1h/4h` (80 predictions). The bugs affect both pairs but the evidence may differ in severity.
**Why it happens:** Both pairs use the same `fit_xgb` function and the same `LivePredictor` + `PredictionValidator` path.
**How to avoid:** Run all diagnostic queries with `GROUP BY freq, horizon`.
**Warning signs:** Only checking one pair and concluding the other is fine.

### Pitfall 4: Treating this as an implementation phase
**What goes wrong:** The phase outcome is a committed ROOT_CAUSE.md document. Writing fix code before the document exists violates DIAG-02.
**Why it happens:** Once the bugs are found, the impulse is to fix them immediately.
**How to avoid:** The plan must explicitly gate on document commit before any fix task. The success criteria states "committed before any fix code was merged."

---

## Code Examples

### Verify Model Objective (Stage 4 boundary)
```python
# Source: direct codebase inspection — src/bitbat/model/train.py L52-56
# The bug is here:
params = {
    "objective": "reg:squarederror",  # BUG: regression, not classification
    "eval_metric": "rmse",
    ...
}
```

```python
# Reproducible verification script
import xgboost as xgb
import json

booster = xgb.Booster()
booster.load_model("models/5m_30m/xgb.json")
cfg = json.loads(booster.save_config())
objective = cfg["learner"]["objective"]["name"]
print(f"Model objective: {objective}")
# Output: reg:squarederror  (confirms bug)
```

### Verify Direction Derivation (Stage 5 — Serving)
```python
# Source: src/bitbat/model/infer.py L86-88
# The bug: no tau threshold applied to derive direction
predicted_direction = "up" if predicted_return > 0 else "down"
# Never predicts "flat". With regression model biased negative, mostly "down".
```

```python
# Diagnosis: count direction bias in live predictions
import sqlite3
conn = sqlite3.connect("data/autonomous.db")
cur = conn.cursor()
cur.execute("""
    SELECT predicted_direction, COUNT(*) as cnt,
           AVG(predicted_return) as avg_return
    FROM prediction_outcomes
    GROUP BY predicted_direction
""")
for row in cur.fetchall():
    print(row)
# Output: ('down', 203, -0.00318), ('up', 65, 0.00289)
# 76% down predictions = regression model biased negative
```

### Verify Validation Corruption (Stage 6 — Realization)
```python
# Source: src/bitbat/autonomous/validator.py L45-46, L225-231
# Bug 1: tau hardcoded to 0.0 (tau param ignored)
self.tau = 0.0
# Bug 2: price lookup returns exact same price for many bars

# Diagnosis query
import sqlite3
conn = sqlite3.connect("data/autonomous.db")
cur = conn.cursor()
cur.execute("SELECT COUNT(*) FROM prediction_outcomes WHERE actual_return = 0.0")
zero_returns = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM prediction_outcomes WHERE actual_return IS NOT NULL")
total_realized = cur.fetchone()[0]
print(f"Zero actual_return: {zero_returns}/{total_realized} = {zero_returns/total_realized:.1%}")
# Output: 179/266 = 67.3%  (confirms price lookup corruption)
```

### Full Accuracy Calculation Verification
```python
# Source: src/bitbat/autonomous/db.py L279-280
# correctness = predicted_direction == actual_direction
# With regression model: predicted = "down" (76%), actual = "flat" (67%) -> wrong

import sqlite3
conn = sqlite3.connect("data/autonomous.db")
cur = conn.cursor()
cur.execute("""
    SELECT predicted_direction, actual_direction, correct, COUNT(*) as cnt
    FROM prediction_outcomes
    WHERE actual_return IS NOT NULL
    GROUP BY predicted_direction, actual_direction, correct
    ORDER BY cnt DESC
""")
for row in cur.fetchall():
    print(row)
# Key output: ('down', 'flat', 0, 134) — 134 wrong because "down" != "flat"
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| No live accuracy tracking | `prediction_outcomes` table with `correct` field | v1.3 | Live accuracy visible but revealed bugs |
| Manual diagnosis | CLI `validate run` + `monitor status` | v1.3 | Partially automated |

**Deprecated/outdated:**
- `self.tau = 0.0` hardcoding in `validator.py`: was a placeholder that should have used the configured `tau` value. The parameter exists on `__init__` but is ignored (line 46 overwrites it).
- `reg:squarederror` for 3-class direction prediction: regression objective has no concept of directional boundaries.

---

## Open Questions

1. **Is the price-lookup zero-return bug a data gap or a logic bug?**
   - What we know: 179/266 realized predictions have `actual_return = 0.0` exactly. The validator uses a 60-minute tolerance window. The `data/raw/prices/` directory should have 5m bars.
   - What's unclear: Whether the price data actually has gaps in the validation window, or whether the lookup logic returns start_price as end_price when no match is found.
   - Recommendation: The diagnostic task should check price data coverage for the prediction timestamps: `data/raw/prices/**/*_5m.parquet` date range vs prediction dates. Low priority relative to the model objective bug since Phase 30 will reset all data anyway.

2. **Does the `correct` field in pre-existing DB rows need recalculation?**
   - What we know: Phase 30 includes a reset procedure (`data/` + `models/` + `autonomous.db` wipe).
   - What's unclear: Whether the diagnosis phase needs to recalculate historical `correct` values with a fixed validator to measure the "true" counterfactual accuracy.
   - Recommendation: Not required for DIAG-01/DIAG-02. The diagnosis documents the bugs; the reset/fix in Phase 30 starts fresh.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (existing) |
| Config file | `pyproject.toml` (pytest section) |
| Quick run command | `poetry run pytest tests/model/test_train.py tests/autonomous/test_validator.py -x` |
| Full suite command | `poetry run pytest` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|--------------|
| DIAG-01 | CLI command surfaces which pipeline stage failed | smoke | `poetry run bitbat validate run && poetry run bitbat monitor status` | manual-only — CLI output inspection |
| DIAG-02 | ROOT_CAUSE.md committed before fix code | structural | `poetry run pytest tests/docs/test_root_cause_exists.py -x` | ❌ Wave 0 |

**DIAG-01 note:** The "operator can run a CLI command" requirement is fundamentally a manual inspection deliverable. The automated component is: (a) a pytest test that asserts `models/{freq}_{horizon}/xgb.json` objective is `reg:squarederror` (confirming the bug exists), and (b) a pytest test that queries `autonomous.db` and asserts hit_rate < 0.33 (confirming accuracy collapse is real). These serve as the reproducible trace for the document.

### Sampling Rate
- **Per task commit:** `poetry run pytest tests/autonomous/test_validator.py tests/model/test_train.py -x`
- **Per wave merge:** `poetry run pytest`
- **Phase gate:** Full suite green + ROOT_CAUSE.md committed before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/docs/test_root_cause_exists.py` — structural test asserting ROOT_CAUSE.md exists and has required sections (stage table, reproducible trace) — covers DIAG-02
- [ ] `tests/diagnosis/test_pipeline_stage_trace.py` — automated stage-by-stage inspection: assert model objective = "reg:squarederror", assert hit_rate < 0.33 from DB — covers DIAG-01 reproducible trace

*(ROOT_CAUSE.md itself is a planning document, not a test. The tests verify it exists and the bugs it documents are confirmed by automated checks.)*

---

## Sources

### Primary (HIGH confidence)
- Direct code inspection: `/home/eimi/projects/ai-btc-predictor/src/bitbat/model/train.py:53` — `objective: reg:squarederror`
- Direct code inspection: `/home/eimi/projects/ai-btc-predictor/src/bitbat/model/infer.py:86-88` — direction from sign, no tau
- Direct code inspection: `/home/eimi/projects/ai-btc-predictor/src/bitbat/autonomous/validator.py:46` — `self.tau = 0.0` hardcoded
- Live DB query: `autonomous.db` — 38/266 = 14.3% accuracy confirmed
- Live DB query: `autonomous.db` — 179/266 = 67% exact zero `actual_return` values
- Live DB query: `autonomous.db` — 203/268 = 76% predictions labeled "down"
- Model artifact inspection: `xgb.Booster.save_config()` on both `5m_30m/xgb.json` and `1h_4h/xgb.json` — both confirm `reg:squarederror`
- Dataset inspection: `data/features/5m_30m/meta.json` — `label_mode: return_direction`, no `flat` class in training labels

### Secondary (MEDIUM confidence)
- `data/features/5m_30m/dataset.parquet` label distribution: only `up` (4152) and `down` (4162) labels — no `flat` class used in training (tau=0.0 during dataset assembly)

### Tertiary (LOW confidence — needs Phase 30 validation)
- Price lookup zero-return bug: hypothesis that `get_price_at_timestamp` returns `start_price` as `end_price` due to price data gaps in the validation window. Confirmed by symptom (179 zero returns) but root cause of the gap not yet verified.

---

## Metadata

**Confidence breakdown:**
- Model objective bug (DIAG-01 stage 4): HIGH — confirmed by code + artifact inspection
- Serving direction derivation (DIAG-01 stage 5): HIGH — confirmed by code + DB distribution
- Validation tau=0 hardcoding (DIAG-01 stage 6): HIGH — confirmed by code (line 46)
- Price lookup zero-return (DIAG-01 stage 6): MEDIUM — confirmed by symptom, root cause of price gap unverified
- DIAG-02 document structure: HIGH — standard root-cause document pattern, no ambiguity

**Research date:** 2026-03-08
**Valid until:** 2026-04-07 (30 days — stable domain, findings are about existing code)

---

## Discovered Root Cause Summary

For the planner's convenience, the three bugs are:

| # | Stage | File | Line | Bug | Severity |
|---|-------|------|------|-----|----------|
| 1 | Model | `src/bitbat/model/train.py` | 53 | `objective: reg:squarederror` instead of `multi:softprob` — trains on float returns, not class labels | PRIMARY |
| 2 | Serving | `src/bitbat/model/infer.py` | 86 | `direction = "up" if predicted_return > 0 else "down"` — no tau threshold, binary only | SECONDARY |
| 3 | Validation | `src/bitbat/autonomous/validator.py` | 46 | `self.tau = 0.0` hardcoded — overrides constructor param, corrupts flat classification | TERTIARY |

**Compounding effect:** Bug 1 produces a regression model biased toward predicting negative returns (mean = -0.00246). Bug 2 maps this to "down" 76% of the time. Bug 3 (combined with price data gaps) produces 67% "flat" actual outcomes. The intersection of mostly-"down" predictions against mostly-"flat" actuals yields 14% accuracy.
