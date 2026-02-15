# BitBat Project Memory

## Project Structure
- Source: `src/bitbat/` (package renamed from "alpha" to "bitbat")
- Autonomous system: `src/bitbat/autonomous/`
- Tests: `tests/` with `tests/autonomous/` for autonomous system tests
- Scripts: `scripts/` (run_monitoring_agent.py, run_ingestion_service.py, init_autonomous_db.py)
- Deployment: `deployment/` (systemd service files)
- Docs: `docs/` (schema, strategy, completion reports)

## Phase 1 Sessions Status
- SESSION 1: Database foundation (models, schema) — COMPLETE
- SESSION 2: Prediction validator — COMPLETE
- SESSION 3: Autonomous monitoring agent (drift, retraining, alerting) — COMPLETE
- SESSION 4: Continuous data ingestion — COMPLETE

## Phase 2 Sessions Status (GUI Redesign)
- SESSION 1: Configuration presets + simplified home dashboard — COMPLETE
- SESSION 2: Live monitoring (widgets, auto-refresh, activity feed, countdown, System page) — COMPLETE
- SESSION 3: Alerting (Alerts page, alert rules, in-app notifications, mobile CSS) — COMPLETE

## Phase 2 Files Created
- `docs/gui_redesign.md` — strategy doc, terminology translations
- `src/bitbat/gui/__init__.py`, `src/bitbat/gui/presets.py` — Preset dataclass + 3 presets (Conservative/Balanced/Aggressive)
- `src/bitbat/gui/widgets.py` — DB-backed data helpers + Streamlit render helpers (no st import at module level)
- `streamlit/app.py` — REPLACED with simplified home dashboard (auto-refresh, activity feed, countdown)
- `streamlit/pages/1_⚙️_Settings.py` — preset selector with advanced settings
- `streamlit/pages/2_📈_Performance.py` — accuracy, streaks, recent predictions, model info
- `streamlit/pages/3_ℹ️_About.py` — FAQ, plain-English how-it-works, disclaimer
- `streamlit/pages/4_🔧_System.py` — ingestion status, agent status, logs, snapshots
- `streamlit/pages/5_🔔_Alerts.py` — email/Discord/Telegram config, test buttons, alert rules, history
- `streamlit/pages/9_🔬_Pipeline.py` — old technical app preserved for power users
- `streamlit/style.py` — shared CSS with mobile-friendly responsive layout
- `.streamlit/config.toml` — green theme, CORS disabled
- `config/alert_rules.yaml` — alert rules configuration file
- `tests/gui/test_presets.py` — 21 tests
- `tests/gui/test_widgets.py` — 21 tests
- `tests/gui/test_complete_gui.py` — 13 integration tests

## Phase 2 GUI Pattern
- Streamlit multi-page app: `app.py` = home, `pages/` = named pages
- All pages handle missing DB gracefully (never show raw errors)
- Technical jargon translated: freq→Update Frequency, horizon→Forecast Period, tau→Movement Sensitivity
- Data only ever accessed via `widgets.db_query()` which returns [] on any failure
- Auto-refresh via `<meta http-equiv='refresh' content='60'>` in home page

## Key Conventions
- `from __future__ import annotations` at top of every module
- UTC datetime: use `datetime.now(UTC).replace(tzinfo=None)` (NOT `datetime.utcnow()`)
- Write parquet: use `bitbat.io.fs.write_parquet(df, path)` — handles dir creation
- Data contracts: always call `ensure_prices_contract()` / `ensure_news_contract()` before writing
- CLI pattern: `@_cli.group()` then `@group.command()` decorators in `src/bitbat/cli.py`

## SESSION 4 Files Created
- `docs/ingestion_strategy.md`
- `src/bitbat/autonomous/rate_limiter.py` — RateLimiter class, period-based, JSON-persisted
- `src/bitbat/autonomous/price_ingestion.py` — PriceIngestionService (yfinance, date-partitioned)
- `src/bitbat/autonomous/news_ingestion.py` — NewsIngestionService (CryptoCompare free, NewsAPI, Reddit optional)
- `scripts/run_ingestion_service.py` — blocking loop, SIGINT/SIGTERM graceful shutdown
- `deployment/bitbat-ingest.service` — systemd unit
- `docker-compose.yml` — ingest + monitor + UI services
- `tests/autonomous/test_ingestion.py` — 18 unit tests (mocked)
- `tests/autonomous/test_session4_complete.py` — 6 integration tests

## Free News APIs
- CryptoCompare: no key needed, `https://min-api.cryptocompare.com/data/v2/news/`
- NewsAPI: env var `NEWSAPI_KEY`, 100 req/day free, get at newsapi.org
- Reddit (optional): `REDDIT_CLIENT_ID` + `REDDIT_CLIENT_SECRET`, needs `praw` package

## Test Commands
```bash
poetry run pytest tests/autonomous/ -v
poetry run bitbat ingest prices-once
poetry run bitbat ingest news-once
poetry run bitbat ingest status
poetry run python scripts/run_ingestion_service.py
```
