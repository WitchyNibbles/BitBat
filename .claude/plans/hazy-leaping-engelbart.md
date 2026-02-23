# Plan: One-Click Train + Prediction Timeline

## Context

The current Pipeline page (`streamlit/pages/9_Pipeline.py`) requires 8-10 clicks across 7 sub-pages to go from nothing to a trained model making predictions: ingest prices, ingest news, build features, run CV, train model, batch run, realize. The user wants a single "Train" button that does everything, then shows an auto-refreshing prediction timeline chart.

The autonomous system already handles hourly predictions, validation, drift detection, and retraining -- we just need to bridge the gap between "user clicks train" and "autonomous loop is running".

---

## 1. Create `src/bitbat/autonomous/orchestrator.py` — One-click training function

A single `one_click_train()` function that chains existing primitives directly (no subprocess):

```python
def one_click_train(
    preset_name: str = "balanced",
    progress_callback: Callable[[str, float], None] | None = None,
) -> dict[str, Any]:
```

**Steps (reusing existing code):**

| Step | Progress | What | Calls |
|------|----------|------|-------|
| 1 | 0-5% | Load config + preset | `get_runtime_config()`, `get_preset(preset_name)` |
| 2 | 5-25% | Ingest prices (2 years) | `prices_module.fetch_yf(symbol, freq, start)` |
| 3 | 25-40% | Ingest news (if sentiment enabled) | `news_cc_module.fetch(from_dt, to_dt)` |
| 4 | 40-60% | Build features + labels | `build_xy(prices_path, news_path, freq, horizon, tau, ...)` |
| 5 | 60-85% | Train XGBoost model | `fit_xgb(X, y)` — saves to `models/{freq}_{horizon}/xgb.json` |
| 6 | 85-95% | Register model in DB | `db.store_model_version(...)` |
| 7 | 95-100% | Generate first prediction | `LivePredictor(db).predict_latest()` |

Returns dict: `{status, model_version, training_samples, duration_seconds, error?}`

Key details:
- Calls `set_runtime_config()` to write the preset's params to user config YAML
- Uses `prices_module.fetch_yf()` from `bitbat.ingest.prices` (same as Pipeline page)
- Uses `news_cc_module.fetch()` from `bitbat.ingest.news_cryptocompare`
- Catches errors at each step and returns `{"status": "failed", "error": "...", "step": "..."}`
- Initializes DB via `init_database()` from `bitbat.autonomous.models`

---

## 2. Create `src/bitbat/gui/timeline.py` — Prediction timeline chart

Two functions:

### `get_timeline_data(db: AutonomousDB, freq, horizon, limit=168) -> pd.DataFrame`
- Queries `prediction_outcomes` table via `db.get_recent_predictions(realized_only=False)`
- Returns DataFrame with: `timestamp_utc`, `predicted_direction`, `p_up`, `p_down`, `actual_direction`, `correct`

### `render_prediction_timeline(predictions: pd.DataFrame, prices: pd.DataFrame) -> None`
- Uses **Plotly** (`plotly.graph_objects`) for an interactive chart
- **Price line**: BTC close price as a continuous line (left y-axis)
- **Prediction markers** overlaid on the price at each prediction timestamp:
  - Green triangle-up = predicted UP
  - Red triangle-down = predicted DOWN
  - Gray circle = predicted FLAT
  - Solid/bright = correct, dim/faded = wrong, medium = pending (not yet realized)
- Hover tooltip: time, prediction, confidence, result
- Dark theme matching existing charts (`plotly_dark` template)
- Called via `st.plotly_chart(fig, use_container_width=True)`

Price data loaded from ingested parquet via existing `pd.read_parquet()` on `data/raw/prices/btcusd_yf_{freq}.parquet`.

---

## 3. Create `streamlit/pages/0_Quick_Start.py` — Simplified page

Named with `0_` prefix so it appears first in sidebar. Three states via `st.session_state`:

### State: INITIAL (no model trained yet)
- Title: "BitBat Quick Start"
- Preset radio selector (Conservative / Balanced / Aggressive) with description from `preset.to_display()`
- Large "Train Model" primary button

### State: TRAINING (button clicked)
- Progress bar + status text updated by `one_click_train(progress_callback=...)`
- On completion: store result in session state, transition to RUNNING, `st.rerun()`

### State: RUNNING (model trained, predictions active)
- Success banner with training summary (samples, duration)
- **Monitoring thread**: Start `MonitoringAgent.run_once()` in a daemon `threading.Thread` looping every `validation_interval` seconds (default 3600). Store thread reference in `st.session_state`. Check if already running before starting.
- **Auto-refreshing timeline**: `@st.fragment(run_every=60)` that calls `get_timeline_data()` + `render_prediction_timeline()`
- Summary metrics row: Total Predictions, Completed, Correct, Accuracy %
- "Retrain" button to re-run `one_click_train()` with fresh data

### State detection on page load
- If model file exists at `models/{freq}_{horizon}/xgb.json` → skip to RUNNING state
- If monitoring thread is already alive in session_state → show timeline directly

---

## 4. Modify `streamlit/app.py` — Add Quick Start button

In the "Getting Started" section (lines 163-189) and Quick Actions (lines 133-144):
- Add a "Quick Start" button that calls `st.switch_page("pages/0_Quick_Start.py")`
- Replace the bash command instructions with a pointer to the Quick Start page

---

## 5. Tests

### `tests/autonomous/test_orchestrator.py`
- Mock `prices_module.fetch_yf`, `news_cc_module.fetch`, `build_xy`, `fit_xgb`
- Verify `one_click_train()` calls each step in order
- Verify progress_callback is called with increasing fractions
- Verify error at any step returns `{"status": "failed", "step": "..."}`

### `tests/gui/test_timeline.py`
- `test_get_timeline_data`: mock DB, verify DataFrame columns
- `test_render_prediction_timeline`: build synthetic data, verify Plotly figure is produced (check `fig.data` has expected traces)

---

## File Summary

| Action | File |
|--------|------|
| Create | `src/bitbat/autonomous/orchestrator.py` |
| Create | `src/bitbat/gui/timeline.py` |
| Create | `streamlit/pages/0_Quick_Start.py` |
| Create | `tests/autonomous/test_orchestrator.py` |
| Create | `tests/gui/test_timeline.py` |
| Modify | `streamlit/app.py` |

---

## Verification

1. `poetry run ruff check src/ tests/ streamlit/` — lint clean
2. `poetry run ruff format --check src/ tests/ streamlit/` — format clean
3. `poetry run pytest tests/ -q --tb=short -x` — all tests pass
4. Manual: open Streamlit, navigate to Quick Start, select Balanced, click Train → verify progress bar completes → verify timeline appears after first prediction
