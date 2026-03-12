# Codebase Concerns

**Analysis Date:** 2026-02-24

## Tech Debt

**Broad Exception Handling:**
- Issue: Multiple modules catch bare `Exception` without specific handling, masking root causes
- Files: `src/bitbat/ingest/news_gdelt.py` (line 162, 234), `src/bitbat/autonomous/validator.py` (lines 118, 297, 348), `src/bitbat/autonomous/news_ingestion.py`
- Impact: Difficult debugging; silent failures in data ingestion pipelines; hard to distinguish transient network errors from data schema violations
- Fix approach: Replace bare `except Exception` with specific exception types (e.g., `ValueError`, `RequestException`); add structured logging with error context; consider adding error telemetry for production environments

**Temporal Helper Function Duplication:**
- Issue: `_utcnow()` function defined identically in at least 3 modules (db.py, validator.py, and others)
- Files: `src/bitbat/autonomous/db.py` (line 22), `src/bitbat/autonomous/validator.py` (line 20)
- Impact: Code duplication; inconsistency if one is updated without others; harder to maintain timezone handling logic
- Fix approach: Extract to `bitbat.timealign.calendar` as a shared utility; import everywhere needed

**Price Data Fetch Tolerance Hardcoded:**
- Issue: 60-minute tolerance window for matching prediction timestamps to prices is hardcoded in multiple places
- Files: `src/bitbat/autonomous/validator.py` (lines 237, 251)
- Impact: Cannot adjust tolerance without code changes; may miss valid predictions with minor timing gaps
- Fix approach: Move to configuration; consider making tolerance configurable per frequency

**News Ingestion Partition Strategy Opaque:**
- Issue: GDELT/CryptoCompare news ingestion uses year/month partitions but rationale and implications unclear
- Files: `src/bitbat/ingest/news_gdelt.py` (lines 312-323), `src/bitbat/ingest/news_cryptocompare.py`
- Impact: Difficult to query cross-month news; potential inefficiency in large datasets
- Fix approach: Document partitioning strategy in code comments; consider whether hourly or daily partitions would be more efficient

**Session Management in GDELT Fetch:**
- Issue: Session lifecycle management in `_fetch_chunk` is implicit and error-prone
- Files: `src/bitbat/ingest/news_gdelt.py` (lines 262-293)
- Impact: Potential resource leaks if exception occurs during retry loop; cleanup only happens at top-level scope
- Fix approach: Use context managers consistently; wrap retry logic in try/finally to ensure session cleanup

## Known Bugs

**Price Validation Anomaly Detection Weak:**
- Symptoms: Returns > 50% trigger a warning but continue processing instead of failing
- Files: `src/bitbat/autonomous/validator.py` (lines 272-279)
- Trigger: Large intraday price movements or data corruption
- Workaround: Manual inspection of logs for WARNING level messages; no automatic failure handling
- Impact: May accept unreliable realized returns as ground truth; corrupted price data silently contaminates performance metrics

**Prediction Validation Duplicate Source Risk:**
- Symptoms: `fetch_price_data()` loads all `.parquet` files in prices directory; if multiple files cover same timestamp, behavior is "last wins"
- Files: `src/bitbat/autonomous/validator.py` (lines 107-109, 142-147)
- Trigger: Manual addition of price files or overlapping ingest windows
- Workaround: Operator must clean up duplicates manually before validation runs
- Impact: Incorrect realized returns if operator accidentally ingests overlapping price ranges

**Sign-Based Correctness Logic Fragile:**
- Symptoms: Validation uses sign comparison (pred_sign == actual_sign) as correctness metric, not directional match
- Files: `src/bitbat/autonomous/validator.py` (lines 264-270)
- Trigger: Very small actual returns near zero cause sign mismatches
- Workaround: None; requires code change to add tolerance
- Impact: Hit rate metrics unreliable near inflection points; small movements incorrectly counted as failures

## Security Considerations

**SQLite Database Unencrypted:**
- Risk: Autonomous monitoring system stores all predictions, performance snapshots, and model versions in plaintext SQLite
- Files: `src/bitbat/autonomous/db.py` (line 29), `src/bitbat/autonomous/models.py`
- Current mitigation: SQLite typically stored locally; no network exposure documented
- Recommendations: If deployed to shared environments, encrypt database at rest; consider using encrypted database backend (e.g., sqlcipher); audit file permissions on `data/autonomous.db`

**External API Rate Limit Handling Permissive:**
- Risk: GDELT rate limiting (429) and server errors (5xx) retried with exponential backoff but no global circuit breaker
- Files: `src/bitbat/ingest/news_gdelt.py` (lines 122-155)
- Current mitigation: Per-request retry limit (3 by default); individual request timeout (30s)
- Recommendations: Add global request quota; implement circuit breaker pattern to fail fast if API is degraded; monitor retry frequency per run

**Validation Tolerance Allows Stale Prices:**
- Risk: 60-minute tolerance for price matching means predictions validated on prices up to 1 hour delayed
- Files: `src/bitbat/autonomous/validator.py` (lines 172, 237, 251)
- Current mitigation: Warnings logged for matches > 5 minutes away
- Recommendations: Enforce stricter tolerances for real-time deployment; add alerting if price staleness exceeds threshold

## Performance Bottlenecks

**Price Fetching Unoptimized:**
- Problem: Validator scans entire prices directory, reads all parquets into memory, then filters
- Files: `src/bitbat/autonomous/validator.py` (lines 107-147)
- Cause: No indexing; full table scan; naive string matching for frequency glob
- Improvement path: Add timestamp indexes to parquet metadata; implement binary search for time windows; consider partitioning by date; use DuckDB for efficient filtering without loading all data

**News Ingestion Duplicate Key Check Expensive:**
- Problem: `drop_duplicates(subset=["url"])` on merged DataFrames forces full table compare
- Files: `src/bitbat/ingest/news_gdelt.py` (lines 299, 307)
- Cause: O(n) duplicate detection; repeats on every fetch
- Improvement path: Use set-based deduplication for new articles; track ingested URLs in fast lookup table; consider bloom filter

**Autonomous Validation Loop Synchronous:**
- Problem: Prediction validator runs sequentially; fetches price data once, then validates each prediction one-by-one
- Files: `src/bitbat/autonomous/validator.py` (lines 334-356)
- Cause: No parallelization; single-threaded database commits
- Improvement path: Batch database operations; consider async/await for I/O operations; parallelize price fetch and validation if thread-safe

**CLI Model Loading No Caching:**
- Problem: `load_model()` called on every command invocation; no in-process caching
- Files: `src/bitbat/cli.py` (large file, 1079 lines; model load scattered throughout)
- Cause: Model loaded fresh each time; xgboost JSON parsing not cached
- Improvement path: Implement LRU cache for model loading; add cache invalidation on file modification; consider shared model registry

## Fragile Areas

**Feature Dataset Assembly Leakage Risk:**
- Files: `src/bitbat/dataset/build.py` (line 229)
- Why fragile: Forward-return computation and embargo bar enforcement critical to prevent future leakage; any change to timealign logic breaks guarantees
- Safe modification: All changes to `bitbat/timealign/` must be accompanied by running full test suite including `tests/features/test_leakage.py`; PR-AUC guardrail test must pass
- Test coverage: `tests/features/test_leakage.py` specifically tests embargo correctness; `tests/dataset/` validates split structure

**Autonomous Retraining Decision Logic:**
- Files: `src/bitbat/autonomous/retrainer.py` (line 242), `src/bitbat/autonomous/orchestrator.py` (line 255)
- Why fragile: Drift detection triggers retraining; if threshold or metrics change, model may retrain unnecessarily (resource waste) or not retrain when needed (performance degradation)
- Safe modification: Changes to drift thresholds must be tested against historical performance metrics; add simulation mode before deploying to production
- Test coverage: `tests/autonomous/test_metrics.py` (line 222) validates metric computation but lacks integration tests for full drift scenario

**Contract Enforcement Layer:**
- Files: `src/bitbat/contracts.py` (line 160)
- Why fragile: Data contracts are strict; any upstream schema change breaks downstream pipelines
- Safe modification: Schema changes must be propagated through all consumers (features, labeling, model); add migration logic for backward compatibility
- Test coverage: `tests/contracts/test_contracts.py` validates individual contracts but lacks end-to-end pipeline validation

**Validation Timestamp Normalization:**
- Files: `src/bitbat/autonomous/validator.py` (lines 24-28, 149-150)
- Why fragile: Multiple timestamp normalization paths (pandas Timestamp conversion, UTC stripping); inconsistency causes off-by-one errors in validation
- Safe modification: Centralize all timezone handling to `bitbat.timealign.calendar`; add strict type hints
- Test coverage: Unit tests for `_normalize_timestamp` exist but lack fixtures covering daylight saving time edge cases

## Scaling Limits

**SQLite Autonomous Database Single-File Bottleneck:**
- Current capacity: Works well < 1M prediction rows; SQLite performance degrades with large datasets
- Limit: Expected to hit performance walls at 10M+ rows; concurrent write contention under high-frequency prediction scenarios
- Scaling path: Migrate to PostgreSQL for distributed deployments; add connection pooling; implement time-series partitioning

**News Ingestion Partition Explosion:**
- Current capacity: Year/month partitions create ~120 partitions per year
- Limit: If ingesting multiple years or sources, can lead to thousands of small files; query planning overhead increases
- Scaling path: Consolidate partitions into quarterly or yearly buckets; implement rolling window cleanup for old data

**Price Data Parquet Scan Full Table:**
- Current capacity: Validator scans all price files; workable < 100GB
- Limit: With multiple assets or high-frequency data, scan time becomes prohibitive
- Scaling path: Implement columnar filtering at read time; use DuckDB or Polars for predicate pushdown; add timestamp indexes

## Dependencies at Risk

**GDELT API Dependency Opacity:**
- Risk: GDELT endpoint documentation sparse; response schema inferred from code; breaking changes likely to go undetected
- Files: `src/bitbat/ingest/news_gdelt.py` (lines 191-193)
- Impact: API response format change silently produces empty results or schema violations
- Migration plan: Add robust response validation; log full response on parse failure; consider fallback to CryptoCompare if GDELT fails

**XGBoost Model Binary Format Lock-In:**
- Risk: Models persisted as XGBoost JSON; version mismatch between training and inference causes loading failures
- Files: `src/bitbat/model/persist.py`, `src/bitbat/model/train.py`
- Impact: Cannot easily upgrade XGBoost without retraining all models; inference may fail if library version changes
- Migration plan: Add model versioning metadata; implement compatibility wrapper; consider ONNX format for portability

**Requests Library Optional Import:**
- Risk: GDELT news fetcher requires requests package but imports deferred, causing runtime failures
- Files: `src/bitbat/ingest/news_gdelt.py` (lines 15-18)
- Impact: Delayed error discovery; users discover missing dependency only when running news ingestion
- Migration plan: Add explicit dependency check at module load time or CLI startup; document all optional dependencies clearly

## Missing Critical Features

**No Prediction Confidence Intervals:**
- Problem: Model outputs point predictions; no uncertainty quantification
- Blocks: Cannot assess prediction reliability; difficult to set risk limits
- Risk: High-confidence incorrect predictions treated same as low-confidence ones

**No Model Performance Degradation Alerts:**
- Problem: Retraining only triggered by drift; no proactive alerts for subtle performance decay
- Blocks: May trade with degraded model for extended periods before drift threshold triggers
- Risk: Slow model rot goes unnoticed; opportunity cost from suboptimal decisions

**No Backtesting on Autonomous Predictions:**
- Problem: Feature engineering validated via walk-forward CV; autonomous batch predictions not validated against backtest engine
- Blocks: Cannot confirm realized performance matches backtest expectations
- Risk: Model performs well in backtest but poorly in production due to data handling differences

## Test Coverage Gaps

**Autonomous Drift Detection Untested:**
- What's not tested: Full drift detection workflow (metrics computation → threshold comparison → retraining decision)
- Files: `src/bitbat/autonomous/retrainer.py`, `src/bitbat/autonomous/drift.py`, `src/bitbat/autonomous/orchestrator.py`
- Risk: Drift detection logic changes silently break retraining triggers; models may retrain incorrectly or not at all
- Priority: High - drift decision is critical to production system health

**Validation Timestamp Edge Cases Untested:**
- What's not tested: Daylight saving time transitions, leap seconds, year boundaries
- Files: `src/bitbat/autonomous/validator.py` (lines 24-28)
- Risk: Validation fails or produces incorrect results during DST/year-end periods
- Priority: High - timezone bugs are silent and only appear seasonally

**GDELT API Failure Modes Undertested:**
- What's not tested: Malformed JSON responses, connection timeouts, partial response bodies
- Files: `src/bitbat/ingest/news_gdelt.py` (lines 160-188)
- Risk: Edge case API responses cause unexpected failures or data corruption
- Priority: Medium - external API dependencies less controlled

**Autonomous Database Concurrent Access Untested:**
- What's not tested: Multiple readers/writers to SQLite simultaneously; commit/rollback scenarios
- Files: `src/bitbat/autonomous/db.py` (lines 40-51)
- Risk: Data races; corrupt transaction state under concurrent load
- Priority: Medium - only impacts if scaling to concurrent ingestion

**Feature Contract Enforcement Gaps:**
- What's not tested: All feature combinations (all toggles on/off); contract violations caught by ensure_feature_contract
- Files: `src/bitbat/contracts.py` (lines 79+), `src/bitbat/dataset/build.py`
- Risk: New feature additions bypass contract checks; downstream models receive unexpected columns
- Priority: Medium - contract layer designed to catch this but needs comprehensive test matrix

---

*Concerns audit: 2026-02-24*
