# Home Price Chart Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the Home prediction card with a BTC price vs predicted-price chart backed by a timeline API.

**Architecture:** Add a timeline-oriented API response under prediction routes that merges prediction rows with aligned actual BTC closes. Then replace the Home card with a chart component that renders actual price, predicted price when present, and direction markers with compact status copy.

**Tech Stack:** FastAPI, Pydantic, pandas, React 19, TypeScript, Recharts, pytest

---

### Task 1: Add timeline API contract

**Files:**
- Modify: `src/bitbat/api/schemas.py`
- Modify: `src/bitbat/api/routes/predictions.py`
- Test: `tests/api/test_predictions.py`

**Step 1: Write the failing test**

Add a test that calls the new prediction timeline endpoint and asserts it returns actual BTC price plus prediction metadata.

**Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/api/test_predictions.py -q`

**Step 3: Write minimal implementation**

Add response models and a route that:
- loads recent prediction rows
- loads aligned BTC closes from ingested prices
- returns compact chart points

**Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/api/test_predictions.py -q`

**Step 5: Commit**

```bash
git add src/bitbat/api/schemas.py src/bitbat/api/routes/predictions.py tests/api/test_predictions.py
git commit -m "feat: add home prediction timeline api"
```

### Task 2: Build the Home chart component

**Files:**
- Create: `dashboard/src/components/PredictionPriceChart.tsx`
- Create: `dashboard/src/components/PredictionPriceChart.module.css`
- Modify: `dashboard/src/api/client.ts`
- Modify: `dashboard/src/pages/Home.tsx`
- Modify: `dashboard/src/pages/Home.module.css`

**Step 1: Write the failing integration shape**

Use the new client types and Home page wiring so the app cannot compile until the chart component exists.

**Step 2: Run build to verify it fails**

Run: `npm run build`

**Step 3: Write minimal implementation**

Implement a compact chart card that:
- draws actual BTC price
- overlays predicted price when present
- shows direction markers
- shows latest call, conviction, and recent realized accuracy
- explains missing predicted-price coverage when needed

**Step 4: Run build to verify it passes**

Run: `npm run build`

**Step 5: Commit**

```bash
git add dashboard/src/components/PredictionPriceChart.tsx dashboard/src/components/PredictionPriceChart.module.css dashboard/src/api/client.ts dashboard/src/pages/Home.tsx dashboard/src/pages/Home.module.css
git commit -m "feat: replace home crystal ball with price chart"
```

### Task 3: Final verification

**Files:**
- Verify touched files only

**Step 1: Run backend tests**

Run: `poetry run pytest tests/api/test_predictions.py tests/autonomous/test_metrics.py tests/autonomous/test_drift.py`

**Step 2: Run dashboard build**

Run: `npm run build`

**Step 3: Note known issues**

Record any unrelated lint failures already present in the dashboard.
