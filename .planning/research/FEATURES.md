# Feature Research

**Domain:** Brownfield reliability + timeline UX for prediction monitoring
**Researched:** 2026-02-24
**Confidence:** HIGH

## Feature Landscape

### Table Stakes (Users Expect These)

Features users assume exist. Missing these = product feels broken.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Monitor can read/write prediction records without schema errors | Monitoring is the core operational function | MEDIUM | Requires schema compatibility checks + migration safety |
| Prediction timeline renders historical predictions and outcomes | Timeline is a primary GUI analytic surface | MEDIUM | Must handle missing/late realizations gracefully |
| Timeline filters for freq/horizon/date range | Users need focused analysis windows | LOW | Should be deterministic and fast on local datasets |
| Clear confidence and direction context on timeline entries | Prediction without confidence lacks decision value | LOW | Reuse existing predictor outputs when available |
| No deprecation warnings in normal UI interactions | Warning spam reduces trust and obscures real issues | LOW | Replace `use_container_width` with `width=` patterns globally |

### Differentiators (Competitive Advantage)

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Timeline overlays predicted vs realized return deltas | Makes model quality visible at a glance | MEDIUM | Useful for rapid sanity checks by operator |
| Timeline anomaly markers (miss streaks, drift windows, retrain events) | Connects model behavior to operations | MEDIUM | Leverages existing DB snapshots/events |
| Actionable failure banners with next steps | Reduces MTTR when monitor/UI failures occur | LOW | Better than silent fallback behavior |

### Anti-Features (Commonly Requested, Often Problematic)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Automatic DB drop/recreate on any mismatch | Quick apparent fix | Destroys history and can mask compatibility regressions | Add versioned migration and guarded compatibility checks |
| Over-animated real-time timeline redraw | Feels dynamic | Adds noise, CPU load, and rendering fragility | Stable refresh cadence + explicit update controls |
| "Catch-all" exception suppression in monitor loops | Keeps app running | Hides correctness failures and data corruption risk | Fail critical paths loudly with structured errors |

## Feature Dependencies

```
[Schema versioning + compatibility checks]
    └──requires──> [DB migration workflow]
                       └──requires──> [Automated migration/regression tests]

[Improved timeline overlays]
    └──requires──> [Reliable prediction + realized fields in DB]

[Warning-free GUI]
    └──requires──> [Global Streamlit width API migration]
```

### Dependency Notes

- **Timeline improvements require DB correctness first:** broken prediction columns invalidate chart data.
- **Migration safety requires test coverage:** schema fixes without regression tests re-break quickly.
- **Warning cleanup is independent but should be phased with UI verification tests.**

## MVP Definition

### Launch With (v1)

- [ ] Monitor runs without `predicted_price`/schema runtime failures (D1).
- [ ] Prediction timeline renders correctly with filters and confidence/outcome context (D2).
- [ ] All GUI pages use `width=` APIs; no `use_container_width` warnings remain (D3).
- [ ] Tests assert schema compatibility and timeline data contract.

### Add After Validation (v1.x)

- [ ] Timeline anomaly annotations tied to drift/retraining events.
- [ ] Enhanced operator diagnostics for monitor cycle failures.

### Future Consideration (v2+)

- [ ] Multi-source timeline drill-down and comparative model views.
- [ ] Rich report export for timeline segments.

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| DB schema compatibility + migration | HIGH | MEDIUM | P1 |
| Timeline correctness + filter UX | HIGH | MEDIUM | P1 |
| Streamlit width API migration | HIGH | LOW | P1 |
| Timeline anomaly annotations | MEDIUM | MEDIUM | P2 |
| Export/reporting workflows | LOW | MEDIUM | P3 |

**Priority key:**
- P1: Must have for launch
- P2: Should have, add when possible
- P3: Nice to have, future consideration

## Competitor Feature Analysis

| Feature | Typical Ops Dashboard | Typical Quant Tooling UI | Our Approach |
|---------|-----------------------|---------------------------|--------------|
| Reliability telemetry | Health/error panels | Logs + metrics views | Integrate monitor stability + timeline correctness checks |
| Timeline visualization | Basic charting | Advanced overlays | Deliver practical overlays (predicted vs realized + confidence) first |
| Warning/deprecation hygiene | Usually strict in mature products | Varies by team maturity | Treat warning-free UI as a first-class acceptance gate |

## Sources

- `.planning/PROJECT.md`
- `.planning/codebase/CONCERNS.md`
- `.planning/codebase/ARCHITECTURE.md`
- `streamlit/app.py` and `streamlit/pages/`
- `src/bitbat/autonomous/*`, `src/bitbat/api/*`

---
*Feature research for: BitBat reliability and timeline enhancement*
*Researched: 2026-02-24*
