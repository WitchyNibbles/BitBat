---
status: resolved
trigger: "Two related issues: 1) Retraining fails with 'Not enough samples for configured windows: 8371 < 17280' 2) Predictions showing very low confidence (<5%)"
created: 2026-03-01T00:00:00
updated: 2026-03-01T00:10:00
---

## Current Focus

hypothesis: CONFIRMED and FIXED
test: All 608 tests pass. Manual verification of directional_confidence produces sensible values.
expecting: N/A
next_action: Archive session

## Symptoms

expected: Retraining should complete successfully. Predictions should show reasonable confidence levels (higher than 5%).
actual: Retraining fails with ValueError about insufficient samples (8371 < 17280). Predictions show <5% confidence.
errors: ValueError: Not enough samples for configured windows: 8371 < 17280
reproduction: Occurs during autonomous retraining cycle in continuous_trainer.py line 186
started: After v1.4 config change (freq 1h->5m, horizon 4h->30m)

## Eliminated

## Evidence

- timestamp: 2026-03-01T00:01:00
  checked: default.yaml configuration
  found: freq=5m, horizon=30m, rolling_window_bars=17280, train_window_bars=13824, backtest_window_bars=3456
  implication: 17280 5-min bars = 60 days of continuous data. With 8371 bars available, that's only ~29 days.

- timestamp: 2026-03-01T00:02:00
  checked: continuous_trainer.py _do_retrain method
  found: Line 184-189 computes required_samples = train_window_bars + backtest_window_bars = 13824 + 3456 = 17280. Error at line 186 when len(features) < required_samples.
  implication: The window bars are absolute counts, not time-based. Need to be proportional to available data.

- timestamp: 2026-03-01T00:03:00
  checked: git history (commits 384ab0f and 8be39d4)
  found: Commit 384ab0f changed freq from 1h to 5m and added rolling_window_bars=17280 (60 days of 5m bars). Then 8be39d4 added train_window_bars=13824 and backtest_window_bars=3456. Only ~29 days of 5m data is actually available.
  implication: Window sizes assume more data than exists. Need to reduce to fit available ~29 days.

- timestamp: 2026-03-01T00:04:00
  checked: cli.py batch_run (line 1353) and predictor.py predict_latest (line 336)
  found: Both store_prediction calls omit p_up and p_down parameters, which default to 0.0. The model is regression-based (predicts return magnitude), not classification-based (no class probabilities).
  implication: Dashboard computes confidence = max(p_up, p_down) = max(0.0, 0.0) = 0.0, showing 0% confidence.

- timestamp: 2026-03-01T00:05:00
  checked: gui/widgets.py (line 213-219) and gui/timeline.py (line 153-156)
  found: Both compute confidence from max(p_up, p_down). When both are 0.0, confidence is 0%.
  implication: Confirms low confidence is due to missing p_up/p_down computation.

## Resolution

root_cause: |
  ISSUE 1 (sample shortage): The continuous_training window sizes in default.yaml
  (rolling_window_bars=17280, train_window_bars=13824, backtest_window_bars=3456)
  assumed 60 days of 5-minute data would be available, but only ~29 days existed
  (8371 bars). This was introduced in commit 384ab0f when freq changed from 1h to
  5m without reducing the window sizes proportionally to available data.

  ISSUE 2 (low confidence): After migrating to regression-based predictions (commit
  384ab0f), both prediction paths (CLI batch_run and autonomous predictor) store
  predictions without computing directional probabilities (p_up/p_down default to
  0.0). The dashboard derives confidence as max(p_up, p_down), yielding 0%.

fix: |
  1. Reduced continuous_training window sizes in default.yaml to 14 days of 5m bars:
     rolling_window_bars: 4032, train_window_bars: 3024, backtest_window_bars: 1008
     (requires 4032 bars, well within the available 8371)

  2. Added directional_confidence() to model/infer.py -- a sigmoid mapping that
     converts predicted_return magnitude into p_up/p_down probabilities using the
     tau threshold as a scaling reference. Updated predict_bar() to include p_up
     and p_down in its output.

  3. Wired p_up/p_down through both prediction storage paths:
     - cli.py batch_run: passes tau from config, forwards p_up/p_down to store_prediction
     - autonomous/predictor.py: passes tau from config, forwards p_up/p_down to store_prediction

verification: |
  - All 608 tests pass (no regressions)
  - Lint passes on all changed Python files (pre-existing lint issue on cli.py:1498 is unrelated)
  - Type checking passes on changed files
  - Manual verification: directional_confidence produces sensible confidence values
    (50% at zero return, 73% at 0.5*tau, 88% at tau, 98% at 2*tau)
  - predict_bar now returns p_up and p_down in its output dict
  - New window sizes (4032 required) fit within available data (8371 bars)

files_changed:
  - src/bitbat/config/default.yaml
  - src/bitbat/model/infer.py
  - src/bitbat/cli.py
  - src/bitbat/autonomous/predictor.py
