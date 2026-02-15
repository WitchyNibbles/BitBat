# GUI Redesign Strategy

## Goals
1. Remove all technical jargon (freq, horizon, tau, p_up, p_down)
2. Provide preset configurations for non-technical users
3. Show live system status and predictions
4. Make one-click operations where possible
5. Clear visual feedback at every step

## User Personas

**Beginner User:**
- Wants: Simple "start" button, clear results
- Needs: Clear status, plain English, getting-started guide
- Avoids: Technical configuration

**Intermediate User:**
- Wants: Some control over settings
- Needs: Preset options with plain-language explanations
- Comfortable with: Basic trading concepts

**Advanced User:**
- Wants: Full customization, raw access to pipeline
- Needs: Access to all parameters via Advanced Pipeline page
- Comfortable with: Technical details (freq, tau, horizon)

## Terminology Translation

| Old (Technical)  | New (User-Friendly)         |
|------------------|-----------------------------|
| freq             | Update Frequency            |
| horizon          | Forecast Period             |
| tau              | Movement Sensitivity        |
| p_up / p_down    | Confidence Level            |
| hit_rate         | Accuracy / Correct Rate     |
| sharpe_ratio     | Risk-Adjusted Performance   |
| cv_score         | Model Quality Score         |
| enter_threshold  | Confidence Required         |
| realized_label   | Actual Outcome              |

## Preset System

### Conservative 🛡️
- Fewer, higher-confidence predictions
- Best for: Risk-averse users, long-term holders
- horizon=24h, tau=0.02, enter_threshold=0.75

### Balanced ⚖️ (Default)
- Good mix of frequency and accuracy
- Best for: Most users, recommended starting point
- horizon=4h, tau=0.01, enter_threshold=0.65

### Aggressive 🚀
- More frequent predictions, higher risk
- Best for: Active traders, higher risk tolerance
- horizon=1h, tau=0.005, enter_threshold=0.55

## Page Structure

### Home Dashboard (app.py)
- Large prediction display ("Bitcoin will likely go UP")
- Confidence meter (visual progress bar)
- System status (🟢 Active / 🟡 Idle / ⚪ Not Started)
- Quick stats (accuracy, total predictions)
- Quick action buttons
- Getting started guide (shown when no data)
- Auto-refresh every 60 seconds (Session 2)
- Recent activity feed (Session 2)
- Countdown to next prediction (Session 2)

### Settings Page (pages/1_⚙️_Settings.py)
- Preset selector (3 large buttons with icons)
- Visual indication of selected preset
- Plain-language display of current settings
- Advanced toggle (shows technical params for power users)
- Save/Reset buttons
- Help section with guidance

### Performance Page (pages/2_📈_Performance.py)
- Overall stats (total, correct, accuracy %)
- Rolling accuracy chart
- Recent predictions table (plain language)
- Model version info (Session 2)
- Win/loss streak display

### About Page (pages/3_ℹ️_About.py)
- How it works (plain English)
- FAQ
- Risk disclaimer
- Links to advanced pipeline

### System Health Page (pages/4_🔧_System.py) [Session 2]
- Ingestion service status
- Monitoring agent status
- Recent system events log
- Data freshness indicators

### Alerts Page (pages/5_🔔_Alerts.py) [Session 3]
- Email/Discord/Telegram alert configuration
- Test alert buttons
- Alert history
- Custom alert rules editor

### Advanced Pipeline Page (pages/9_🔬_Pipeline.py)
- Full technical interface (kept for power users)
- All original ingest/features/model/predict/backtest/monitor tabs

## Design Principles
- Always handle missing data gracefully (show "No data yet" not errors)
- Never expose raw database errors to users
- All metrics displayed in plain language with context
- Mobile-friendly layout (simple columns, touch-friendly buttons)
