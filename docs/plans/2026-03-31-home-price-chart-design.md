# Home Price Chart Design

**Date:** 2026-03-31

**Goal:** Replace the Home-page crystal ball card with a clearer BTC price comparison chart that shows actual BTC price, predicted price when available, and prediction direction markers.

## Problem

The current Home prediction card compresses too much meaning into a single label and percentage. Users cannot see how the model call relates to real BTC price movement, and when the runtime emits mostly `flat` predictions or omits `predicted_price`, the UI makes it hard to tell whether the issue is model behavior or presentation.

## Approved Direction

Replace the existing Home prediction card with a compact comparison chart:

- Actual BTC close is the dominant line.
- Predicted price is rendered as a second line when available.
- Prediction timestamps are marked on the actual-price line with up/down/flat semantics.
- The card includes a compact legend and summary strip for latest call, conviction, and recent realized accuracy.
- Missing predicted-price coverage is called out explicitly instead of being hidden.

## Data Shape

The frontend needs a timeline payload that combines:

- `timestamp_utc`
- `actual_price`
- `predicted_price`
- `predicted_direction`
- `confidence`
- `correct`
- `is_realized`

The existing prediction history endpoint does not provide an actual BTC price series suitable for charting, so the API must expose a timeline-oriented response assembled from prediction rows plus aligned close prices from ingested price data.

## UX Notes

- Keep the dashboard’s dark, ceremonial style rather than dropping in a generic finance widget.
- The actual BTC line should feel primary and readable.
- The predicted-price line should be visible but secondary.
- Flat predictions should remain visible instead of being treated as absent.
- If predicted prices are missing for the current model, the card should still tell a coherent story via actual price plus markers.

## Verification

- API tests for the new timeline response contract.
- Home page updated to consume the new endpoint.
- Dashboard production build passes.
