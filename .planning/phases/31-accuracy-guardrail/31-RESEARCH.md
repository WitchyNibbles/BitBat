# Phase 31: Accuracy Guardrail - Research

**Researched:** 2026-03-08
**Domain:** Autonomous monitor agent — accuracy alerting hook
**Confidence:** HIGH

## Summary

Phase 31 adds a realized-accuracy guardrail to the autonomous monitor agent (`MonitoringAgent`). When the rolling window of realized predictions shows a directional hit rate below a configurable threshold (default 40%), the agent must emit a structured alert via the existing `send_alert` channel. The feature is a narrow, well-contained addition: the data already exists in `PerformanceMetrics.hit_rate()`, the alert channel already exists in `alerting.send_alert()`, the config pattern is already established in `drift.py`, and the `run_once()` method in `agent.py` is the correct insertion point.

No new libraries, no schema changes, no new DB tables are required. The work is a config key, a check function in `agent.py`, a YAML default, and a test that seeds low-accuracy predictions into an in-memory DB and asserts the alert fires.

**Primary recommendation:** Add `accuracy_guardrail` sub-section under `autonomous:` in `default.yaml`, add a `_check_accuracy_guardrail()` method to `MonitoringAgent`, call it from `run_once()` after the metrics block, and write one behavioral test using the established `pytest.mark.behavioral` + `tmp_path` + `AutonomousDB` pattern.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| FIXR-04 | Monitor agent alerts when realized accuracy falls below a configurable threshold (default: 40%) | `PerformanceMetrics.hit_rate()` supplies the observed value; `send_alert()` delivers the alert; `autonomous.drift_detection.*` shows the config pattern; `agent.run_once()` is the insertion point |
</phase_requirements>

## Standard Stack

### Core (all already installed — no new dependencies)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `bitbat.autonomous.metrics.PerformanceMetrics` | existing | Computes `hit_rate()` from realized predictions | Already used in `agent.run_once()` |
| `bitbat.autonomous.alerting.send_alert` | existing | Routes alert to email/Discord/Telegram | Already used for drift alerts |
| `bitbat.config.loader.get_runtime_config / load_config` | existing | Reads config at runtime | Already used in `DriftDetector.__init__` |
| `bitbat.autonomous.db.AutonomousDB` | existing | Fetches realized predictions | Already used in `run_once()` |

### No new dependencies required.

## Architecture Patterns

### Where the guardrail logic lives

The guardrail belongs in `MonitoringAgent` (`src/bitbat/autonomous/agent.py`), mirroring the existing drift alert pattern:

```python
# Current drift alert pattern in run_once() — lines 276-281
if drift_detected:
    send_alert(
        "WARNING",
        f"Drift detected for {self.freq}/{self.horizon}",
        {"reason": drift_reason, "metrics": drift_metrics},
    )
```

The accuracy guardrail follows the same shape: check a condition computed from `metrics`, call `send_alert` with structured details.

### Config pattern (from drift.py lines 30-42)

```python
# How DriftDetector reads thresholds from config — HIGH confidence (direct source read)
config = get_runtime_config() or load_config()
autonomous = config.get("autonomous", {})
drift_cfg = autonomous.get("drift_detection", {})
self.directional_accuracy_threshold = float(
    drift_cfg.get("directional_accuracy_threshold", 0.50)
)
```

The accuracy guardrail adds a parallel sub-section `accuracy_guardrail` under `autonomous:` in `default.yaml`, read the same way.

### YAML config addition

```yaml
# Addition to src/bitbat/config/default.yaml under autonomous:
  accuracy_guardrail:
    enabled: true
    min_predictions_required: 10
    realized_accuracy_threshold: 0.40
    window_days: 30
```

- `realized_accuracy_threshold`: alert fires when `hit_rate < this value` (default 0.40)
- `min_predictions_required`: guard against firing on too few samples (mirrors `drift_detection.min_predictions_required`)
- `window_days`: rolling window length; reuses `drift_detector.window_days` or reads its own key
- `enabled`: allows operators to disable the guardrail without removing config keys

### MonitoringAgent method pattern

```python
# New private method on MonitoringAgent — HIGH confidence (mirrors existing code shape)
def _check_accuracy_guardrail(self, metrics: dict[str, Any]) -> bool:
    """Fire alert if realized accuracy falls below configured threshold."""
    config = get_runtime_config() or load_config()
    autonomous = config.get("autonomous", {})
    guardrail_cfg = autonomous.get("accuracy_guardrail", {})

    if not bool(guardrail_cfg.get("enabled", True)):
        return False

    min_required = int(guardrail_cfg.get("min_predictions_required", 10))
    threshold = float(guardrail_cfg.get("realized_accuracy_threshold", 0.40))

    realized = int(metrics.get("realized_predictions", 0))
    if realized < min_required:
        return False

    observed_accuracy = float(metrics.get("hit_rate", 1.0))
    if observed_accuracy < threshold:
        send_alert(
            "WARNING",
            f"Realized accuracy below threshold for {self.freq}/{self.horizon}",
            {
                "observed_accuracy": observed_accuracy,
                "threshold": threshold,
                "realized_predictions": realized,
                "freq": self.freq,
                "horizon": self.horizon,
            },
        )
        return True
    return False
```

### Insertion point in `run_once()` (agent.py line ~270)

After `metrics` is computed from `PerformanceMetrics(recent_predictions).to_dict()` and before the drift check, add:

```python
accuracy_alert_fired = self._check_accuracy_guardrail(metrics)
```

Include `accuracy_alert_fired` in the returned `result` dict for observability.

### Test pattern (from test_drift.py — HIGH confidence)

```python
# Established pattern: pytest.mark.behavioral, tmp_path, in-memory SQLite, seed DB, assert
pytestmark = pytest.mark.behavioral

def test_accuracy_guardrail_fires_on_low_accuracy(tmp_path: Path) -> None:
    database_url = f"sqlite:///{tmp_path / 'guardrail.db'}"
    init_database(database_url)
    db = AutonomousDB(database_url)

    now = datetime.now(UTC).replace(tzinfo=None)
    with db.session() as session:
        for i in range(20):
            pred = db.store_prediction(
                session=session,
                timestamp_utc=now - timedelta(hours=i + 5),
                predicted_direction="up",
                p_up=0.6, p_down=0.2,
                model_version="v1",
                freq="1h", horizon="4h",
            )
            # All incorrect — 0% accuracy
            db.realize_prediction(
                session=session,
                prediction_id=pred.id,
                actual_return=-0.01,
                actual_direction="down",
            )

    agent = MonitoringAgent(db, "1h", "4h")
    # Override threshold to test isolation
    fired = agent._check_accuracy_guardrail({
        "hit_rate": 0.0,
        "realized_predictions": 20,
    })
    assert fired is True
```

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Alert delivery | Custom email/webhook code | `send_alert()` (alerting.py) | Already handles email, Discord, Telegram routing |
| Accuracy calculation | Custom accuracy counter | `PerformanceMetrics.hit_rate()` | Already validated, used in metrics.to_dict() |
| Config loading | Custom YAML reader | `get_runtime_config() or load_config()` | Handles env override, caching, path resolution |
| DB query for realized preds | Custom SQL | `db.get_recent_predictions(realized_only=True)` | Already used in `run_once()` |

## Common Pitfalls

### Pitfall 1: Firing on insufficient sample count
**What goes wrong:** The guardrail fires after only 1-2 realized predictions, causing false positives during system warmup.
**Why it happens:** The hit_rate of 0 on 1 sample looks like collapse.
**How to avoid:** Gate on `min_predictions_required` (default 10) before computing — same pattern as `drift_detection.min_predictions_required`.

### Pitfall 2: Reading hit_rate vs directional_accuracy
**What goes wrong:** `PerformanceMetrics` has two accuracy metrics: `hit_rate()` (uses `correct` column) and `directional_accuracy()` (uses `sign(predicted_return) vs sign(actual_return)`). After Phase 30 fixes, `predicted_return=None` for 3-class classification, so `directional_accuracy()` returns 0.0 always (empty predicted array). The correct field to read is `hit_rate` from `metrics.to_dict()`.
**Warning signs:** `directional_accuracy` always 0.0 — this is a sign the wrong field is being used.

### Pitfall 3: Duplicate alert fire every cycle
**What goes wrong:** Once accuracy falls below threshold it stays there until model retrains. Alert fires every monitor cycle (every 5 minutes by default), flooding channels.
**How to avoid:** Either: (a) log at WARN level once and skip alert if already fired this window (state-based), or (b) keep it simple and accept repeated alerts since operators can mute them — this is acceptable for the phase scope. Phase 31 success criteria does not require deduplication, so (b) is the simpler path.

### Pitfall 4: Modifying `run_once()` return contract
**What goes wrong:** Adding a new key to the `result` dict in `run_once()` that is not expected by callers (tests, API, Streamlit) breaks the interface.
**How to avoid:** Add `accuracy_alert_fired` as an additive key — existing consumers ignore unknown keys. Do not rename or remove existing keys.

### Pitfall 5: MonitoringAgent constructor runs model preflight
**What goes wrong:** `MonitoringAgent.__init__` calls `_validate_model_preflight()` which checks for `models/{freq}_{horizon}/xgb.json`. Tests that construct `MonitoringAgent` need a real model file to exist or must mock the preflight.
**How to avoid:** In tests, mock `_validate_model_preflight` (and `_validate_schema_preflight`) OR test `_check_accuracy_guardrail` as a standalone function by extracting it or calling it on an already-constructed agent with a patched preflight. The simpler approach: extract the check into a standalone function `check_accuracy_guardrail(metrics, config)` and test that function directly — no MonitoringAgent construction needed.

## Code Examples

### Alert details structure (required by success criterion 4)

```python
# Source: alerting.py + success criteria — details dict must contain:
details = {
    "observed_accuracy": float,    # e.g. 0.28
    "threshold": float,            # e.g. 0.40
    "realized_predictions": int,   # e.g. 45
    "freq": str,
    "horizon": str,
}
```

### How `hit_rate` surfaces in metrics dict

```python
# Source: metrics.py PerformanceMetrics.to_dict() — HIGH confidence (direct source read)
# metrics["hit_rate"] == fraction of realized predictions where prediction.correct is True
# metrics["realized_predictions"] == count of predictions with actual_return is not None
```

### How DriftDetector reads config (verified pattern to mirror)

```python
# Source: drift.py lines 30-42 — HIGH confidence (direct source read)
config = get_runtime_config() or load_config()
autonomous = config.get("autonomous", {})
drift_cfg = autonomous.get("drift_detection", {})
self.directional_accuracy_threshold = float(
    drift_cfg.get("directional_accuracy_threshold", 0.50)
)
```

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|------------------|--------|
| No accuracy guardrail (Phase 30 state) | Explicit threshold check in run_once() | Silent collapse detected within one monitor cycle |
| Only drift detection (MAE, losing streak, drift score) | Plus hit_rate vs configured threshold | Directly monitors the metric operators care about |

## Open Questions

1. **Alert deduplication**
   - What we know: Repeated alerts will fire every cycle once accuracy drops (every 5m by default)
   - What's unclear: Whether operators want flood protection
   - Recommendation: Do not add deduplication in Phase 31 — success criteria do not require it, and YAGNI applies. A log line at INFO level noting the guardrail state each cycle is sufficient.

2. **window_days for accuracy guardrail**
   - What we know: `drift_detector.window_days` already computes `recent_predictions` for the same window
   - What's unclear: Whether accuracy guardrail should use the same window or a separate one
   - Recommendation: Reuse the `recent_predictions` already fetched in `run_once()` and passed into `_check_accuracy_guardrail(metrics)` — no second DB query needed. The `window_days` config key in `accuracy_guardrail` is informational but the actual data comes from the already-computed metrics dict.

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (existing) |
| Config file | `pyproject.toml` (pytest config embedded) |
| Quick run command | `poetry run pytest tests/autonomous/test_accuracy_guardrail.py -x` |
| Full suite command | `poetry run pytest` |

### Phase Requirements to Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| FIXR-04 SC1 | Alert fires when hit_rate < threshold with sufficient realized predictions | behavioral unit | `pytest tests/autonomous/test_accuracy_guardrail.py::test_guardrail_fires_on_low_accuracy -x` | No — Wave 0 |
| FIXR-04 SC2 | Threshold is configurable; custom threshold in config overrides 0.40 default | behavioral unit | `pytest tests/autonomous/test_accuracy_guardrail.py::test_guardrail_respects_custom_threshold -x` | No — Wave 0 |
| FIXR-04 SC3 | Guardrail does NOT fire when sample count < min_predictions_required | behavioral unit | `pytest tests/autonomous/test_accuracy_guardrail.py::test_guardrail_skips_insufficient_samples -x` | No — Wave 0 |
| FIXR-04 SC4 | Alert details dict contains observed_accuracy, threshold, realized_predictions | behavioral unit | `pytest tests/autonomous/test_accuracy_guardrail.py::test_guardrail_alert_details -x` | No — Wave 0 |
| FIXR-04 SC2 | YAML key `autonomous.accuracy_guardrail.realized_accuracy_threshold` exists in default.yaml | structural | `pytest tests/autonomous/test_accuracy_guardrail.py::test_guardrail_config_key_in_default_yaml -x` | No — Wave 0 |

### Sampling Rate

- **Per task commit:** `poetry run pytest tests/autonomous/test_accuracy_guardrail.py -x`
- **Per wave merge:** `poetry run pytest`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `tests/autonomous/test_accuracy_guardrail.py` — covers FIXR-04 (all 5 test cases above)
- [ ] No new conftest.py needed — existing `tests/autonomous/` structure is sufficient
- [ ] No framework install needed — pytest already installed

## Sources

### Primary (HIGH confidence)
- Direct source read: `src/bitbat/autonomous/agent.py` — MonitoringAgent.run_once(), alert pattern
- Direct source read: `src/bitbat/autonomous/drift.py` — config reading pattern, DriftDetector
- Direct source read: `src/bitbat/autonomous/alerting.py` — send_alert() signature and routing
- Direct source read: `src/bitbat/autonomous/metrics.py` — PerformanceMetrics.hit_rate(), to_dict()
- Direct source read: `src/bitbat/config/default.yaml` — existing autonomous.drift_detection keys
- Direct source read: `tests/autonomous/test_drift.py` — test pattern (pytest.mark.behavioral, tmp_path, AutonomousDB)
- Direct source read: `tests/autonomous/test_validator.py` — test pattern, DB seeding

### Secondary (MEDIUM confidence)
- Phase success criteria (from ROADMAP.md) — defines the exact alert fields required

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all components are existing code read directly
- Architecture: HIGH — pattern is a direct copy of drift alert code
- Pitfalls: HIGH — identified from direct source reading (hit_rate vs directional_accuracy distinction is especially important after Phase 30 changes)
- Test patterns: HIGH — test_drift.py and test_validator.py provide exact templates

**Research date:** 2026-03-08
**Valid until:** 2026-04-08 (stable internal code — no external dependencies)
