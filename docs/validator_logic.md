# Prediction Validator Logic

## Purpose
Automatically validate predictions by comparing them to actual price movements after the horizon period has elapsed.

## Core Workflow

### 1. Find Unrealized Predictions
The validator fetches predictions where `actual_return` is null and the prediction time is older than `now - horizon`.

```python
def find_predictions_to_validate() -> list[PredictionOutcome]:
    cutoff_time = now_utc - horizon_delta
    return db.get_unrealized_predictions(
        freq=freq,
        horizon=horizon,
        cutoff_time=cutoff_time,
    )
```

### 2. Fetch Actual Price Data
The validator fetches close prices for a single time window that covers all predictions in the batch.

```python
def fetch_price_data(start_time: datetime, end_time: datetime) -> dict[datetime, float]:
    price_frame = load_price_parquet_files()
    filtered = price_frame[(timestamp_utc >= start_time) & (timestamp_utc <= end_time)]
    return dict(zip(filtered["timestamp_utc"], filtered["close"]))
```

### 3. Determine Actual Direction
The validator calculates actual return and maps it to a direction with the training threshold `tau`.

```python
def classify_direction(actual_return: float, tau: float) -> str:
    if actual_return > tau:
        return "up"
    if actual_return < -tau:
        return "down"
    return "flat"
```

### 4. Update Database
After validation, the validator writes the realized result fields.

```python
def realize_prediction(prediction_id: int, actual_return: float, actual_direction: str) -> None:
    db.realize_prediction(
        prediction_id=prediction_id,
        actual_return=actual_return,
        actual_direction=actual_direction,
    )
```

## Error Handling

### Missing Price Data
- Use exact timestamp if available.
- If exact timestamp is missing, use nearest timestamp within tolerance (default 60 minutes).
- Warn when fallback gap is greater than 5 minutes.
- Skip validation when no price is available within tolerance.

### Data Anomalies
- Flag returns with absolute value greater than 50% as anomalies.
- Store the value, but log warning details for later review.

### Database Errors
- Process predictions independently so one failed write does not stop the full run.
- Capture and return validation errors in the summary.

## Frequency
- Development/testing: run manually with `poetry run bitbat validate run`
- Production: schedule hourly via cron or autonomous loop

## Performance Considerations
- Fetch prices once per batch (not per prediction).
- Derive a single min/max time range from prediction timestamps.
- Add a time buffer to account for timestamp tolerance matching.
- Keep batch logic stateless and idempotent for repeated runs.

## Logging Strategy
- `INFO`: start/end summary and batch counts
- `DEBUG`: per-prediction validation details
- `WARNING`: timestamp gap fallbacks and anomalous returns
- `ERROR`: missing data failures and database update failures

## Testing Strategy
- Unit tests for horizon parsing, timestamp matching, direction classification.
- Unit tests for single prediction validation with mock prices.
- Integration tests for batch validation and database updates.
- CLI test for `bitbat validate run` output and option parsing.
