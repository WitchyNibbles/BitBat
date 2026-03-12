"""Repository-style database access helpers for autonomous monitoring."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import desc, inspect, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from .models import (
    ModelVersion,
    PerformanceSnapshot,
    PredictionOutcome,
    RetrainingEvent,
    SystemLog,
    create_database_engine,
    init_database,
)
from .schema_compat import (
    SchemaAuditReport,
    SchemaCompatibilityError,
    audit_schema_compatibility,
    ensure_schema_compatibility,
    format_missing_columns,
    upgrade_schema_compatibility,
)


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _schema_remediation_text(database_url: str, *, can_auto_upgrade: bool) -> str:
    audit_cmd = (
        "poetry run python scripts/init_autonomous_db.py "
        f'--database-url "{database_url}" --audit'
    )
    if can_auto_upgrade:
        upgrade_cmd = (
            "poetry run python scripts/init_autonomous_db.py "
            f'--database-url "{database_url}" --upgrade'
        )
        return f"Run `{audit_cmd}` then `{upgrade_cmd}`."
    return (
        f"Run `{audit_cmd}`. Blocking non-additive incompatibilities were detected; "
        "use `--force` only if table recreation is acceptable."
    )


def _schema_detail_from_report(report: SchemaAuditReport) -> str:
    missing = format_missing_columns(report) or "unknown"
    return f"Schema incompatible: missing {missing}"


def _duration_sort_key(value: str) -> float:
    text_value = str(value).strip().lower()
    units = {"m": 60.0, "h": 3600.0, "d": 86400.0}
    if not text_value:
        return float("inf")
    unit = text_value[-1]
    factor = units.get(unit)
    if factor is None:
        return float("inf")
    try:
        return float(text_value[:-1]) * factor
    except ValueError:
        return float("inf")


@dataclass(slots=True)
class MonitorDatabaseError(RuntimeError):
    """Structured monitor DB failure with actionable diagnostics."""

    step: str
    detail: str
    remediation: str
    error_class: str
    database_url: str

    def __post_init__(self) -> None:
        RuntimeError.__init__(
            self,
            f"[{self.step}] {self.error_class}: {self.detail} Remediation: {self.remediation}",
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "step": self.step,
            "detail": self.detail,
            "remediation": self.remediation,
            "error_class": self.error_class,
            "database_url": self.database_url,
        }


def classify_monitor_db_error(
    exc: Exception,
    *,
    step: str,
    database_url: str,
    engine: Any | None = None,
) -> MonitorDatabaseError:
    """Map raw DB exceptions to actionable monitor diagnostics."""
    error_class = type(exc).__name__
    raw_detail = str(exc).strip() or repr(exc)
    detail = raw_detail
    remediation = (
        f"Check database availability at `{database_url}` and inspect monitor logs for context."
    )

    if isinstance(exc, SchemaCompatibilityError):
        detail = _schema_detail_from_report(exc.report)
        remediation = _schema_remediation_text(
            database_url,
            can_auto_upgrade=exc.report.can_auto_upgrade,
        )
    else:
        lower = raw_detail.lower()
        report: SchemaAuditReport | None = None
        if (
            "no such column" in lower
            or "prediction_outcomes" in lower
            or "performance_snapshots" in lower
        ):
            try:
                report = audit_schema_compatibility(database_url=database_url, engine=engine)
            except Exception:
                report = None

            if report is not None and not report.is_compatible:
                detail = _schema_detail_from_report(report)
                remediation = _schema_remediation_text(
                    database_url,
                    can_auto_upgrade=report.can_auto_upgrade,
                )
            elif "no such column" in lower:
                detail = f"Runtime query failed: {raw_detail}"
                remediation = _schema_remediation_text(database_url, can_auto_upgrade=True)

    return MonitorDatabaseError(
        step=step,
        detail=detail,
        remediation=remediation,
        error_class=error_class,
        database_url=database_url,
    )


class AutonomousDB:
    """High-level interface for autonomous system database operations."""

    def __init__(
        self,
        database_url: str = "sqlite:///data/autonomous.db",
        *,
        auto_upgrade_schema: bool = True,
        allow_incompatible_schema: bool = False,
    ) -> None:
        self.database_url = database_url
        self.engine = create_database_engine(database_url)
        init_database(database_url, engine=self.engine)
        self.schema_compatibility_status: dict[str, str | int] = {}
        if allow_incompatible_schema:
            self.schema_compatibility_status = {
                "upgrade_state": "legacy_readonly",
                "operations_applied": 0,
                "missing_columns_before": 0,
                "missing_columns_after": 0,
            }
        elif auto_upgrade_schema:
            upgrade_result = upgrade_schema_compatibility(
                database_url=database_url,
                engine=self.engine,
            )
            self.schema_compatibility_status = dict(upgrade_result.status)
            if not upgrade_result.is_compatible:
                raise SchemaCompatibilityError(
                    report=upgrade_result.report_after,
                    database_url=database_url,
                )
        else:
            report = ensure_schema_compatibility(
                database_url=database_url,
                engine=self.engine,
                auto_upgrade=False,
                raise_on_error=True,
            )
            self.schema_compatibility_status = {
                "upgrade_state": "already_compatible",
                "operations_applied": 0,
                "missing_columns_before": report.missing_column_count,
                "missing_columns_after": report.missing_column_count,
            }
        self._session_factory = sessionmaker(
            bind=self.engine,
            autoflush=True,
            autocommit=False,
            expire_on_commit=False,
            future=True,
        )
        self._read_retry_attempts = 3
        self._read_circuit_breaker_seconds = 5
        self._circuit_open_until: datetime | None = None

    @contextmanager
    def session(self) -> Any:
        """Yield a managed session with automatic commit/rollback behavior."""
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def _table_columns(self, table_name: str) -> set[str]:
        with self.engine.connect() as connection:
            inspector = inspect(connection)
            if table_name not in set(inspector.get_table_names()):
                return set()
            return {str(column["name"]) for column in inspector.get_columns(table_name)}

    def _first_available(self, columns: set[str], candidates: tuple[str, ...]) -> str | None:
        for candidate in candidates:
            if candidate in columns:
                return candidate
        return None

    def _is_transient_lock_error(self, exc: Exception) -> bool:
        return "database is locked" in str(exc).lower()

    def _raise_circuit_open(self, step: str) -> MonitorDatabaseError:
        raise MonitorDatabaseError(
            step=step,
            detail="Database temporarily unavailable. Circuit open after repeated lock retries.",
            remediation=(
                "Retry shortly. If the circuit stays open, check database availability and "
                "current lock holders."
            ),
            error_class="CircuitOpen",
            database_url=self.database_url,
        )

    def _run_retryable_read(
        self,
        *,
        step: str,
        operation: Callable[[], Any],
    ) -> Any:
        now = _utcnow()
        if self._circuit_open_until is not None and now < self._circuit_open_until:
            self._raise_circuit_open(step)

        last_error: Exception | None = None
        for _ in range(self._read_retry_attempts):
            try:
                result = operation()
            except OperationalError as exc:
                if not self._is_transient_lock_error(exc):
                    raise
                last_error = exc
                continue
            else:
                self._circuit_open_until = None
                return result

        self._circuit_open_until = _utcnow() + timedelta(seconds=self._read_circuit_breaker_seconds)
        detail = "Database temporarily unavailable. Circuit open after repeated lock retries."
        if last_error is not None and not self._is_transient_lock_error(last_error):
            detail = "Database temporarily unavailable."
        raise MonitorDatabaseError(
            step=step,
            detail=detail,
            remediation=(
                "Retry shortly. If the circuit stays open, check database availability and "
                "current lock holders."
            ),
            error_class=type(last_error).__name__ if last_error is not None else "RuntimeError",
            database_url=self.database_url,
        )

    def _run_read(self, *, step: str, operation: Callable[[], Any]) -> Any:
        try:
            return self._run_retryable_read(step=step, operation=operation)
        except MonitorDatabaseError:
            raise
        except Exception as exc:
            if self._is_transient_lock_error(exc):
                self._circuit_open_until = _utcnow() + timedelta(
                    seconds=self._read_circuit_breaker_seconds
                )
                self._raise_circuit_open(step)
            raise classify_monitor_db_error(
                exc,
                step=step,
                database_url=self.database_url,
                engine=self.engine,
            ) from exc

    def _latest_table_value(
        self,
        *,
        table_name: str,
        candidates: tuple[str, ...],
        filters: dict[str, Any] | None = None,
    ) -> Any | None:
        columns = self._table_columns(table_name)
        selected = self._first_available(columns, candidates)
        if selected is None:
            return None

        sql = f"SELECT {selected} AS value FROM {table_name}"  # noqa: S608
        params: dict[str, Any] = {}
        where_parts: list[str] = []
        for key, value in (filters or {}).items():
            if key not in columns:
                continue
            param_name = f"filter_{key}"
            where_parts.append(f"{key} = :{param_name}")
            params[param_name] = value
        if where_parts:
            sql += f" WHERE {' AND '.join(where_parts)}"
        sql += f" ORDER BY {selected} DESC LIMIT 1"

        with self.engine.connect() as connection:
            return connection.execute(text(sql), params).scalar_one_or_none()

    def list_system_logs(self, *, limit: int, level: str | None = None) -> dict[str, Any]:
        def _read() -> dict[str, Any]:
            columns = self._table_columns("system_logs")
            ts_col = self._first_available(columns, ("timestamp", "created_at"))
            if ts_col is None:
                return {"logs": [], "total": 0}

            service_expr = "service" if "service" in columns else "NULL AS service"
            count_sql = "SELECT COUNT(*) FROM system_logs"
            count_params: dict[str, Any] = {}
            if level is not None:
                count_sql += " WHERE level = :level"
                count_params["level"] = level.upper()

            select_sql = (
                f"SELECT {ts_col} AS timestamp, level, message, {service_expr} "  # noqa: S608
                "FROM system_logs"
            )
            select_params: dict[str, Any] = {"limit": limit}
            if level is not None:
                select_sql += " WHERE level = :level"
                select_params["level"] = level.upper()
            select_sql += f" ORDER BY {ts_col} DESC LIMIT :limit"

            with self.engine.connect() as connection:
                total = int(connection.execute(text(count_sql), count_params).scalar_one())
                rows = [
                    dict(row)
                    for row in connection.execute(text(select_sql), select_params).mappings().all()
                ]
            return {"logs": rows, "total": total}

        return self._run_read(step="system.logs", operation=_read)

    def list_retraining_events(self, *, limit: int) -> dict[str, Any]:
        def _read() -> dict[str, Any]:
            columns = self._table_columns("retraining_events")
            if not columns:
                return {"events": [], "total": 0}

            def _col_or_null(name: str) -> str:
                return name if name in columns else f"NULL AS {name}"

            select_cols = [
                "id" if "id" in columns else "rowid AS id",
                _col_or_null("started_at"),
                _col_or_null("trigger_reason"),
                _col_or_null("status"),
                _col_or_null("old_model_version"),
                _col_or_null("new_model_version"),
                _col_or_null("cv_improvement"),
                _col_or_null("training_duration_seconds"),
            ]
            order_col = self._first_available(columns, ("started_at", "id"))
            sql = (
                f"SELECT {', '.join(select_cols)} FROM retraining_events"  # noqa: S608
            )
            if order_col is not None:
                sql += f" ORDER BY {order_col} DESC"
            sql += " LIMIT :limit"

            with self.engine.connect() as connection:
                total = int(
                    connection.execute(text("SELECT COUNT(*) FROM retraining_events")).scalar_one()
                )
                rows = [
                    dict(row)
                    for row in connection.execute(text(sql), {"limit": limit}).mappings().all()
                ]
            return {"events": rows, "total": total}

        return self._run_read(step="system.retraining_events", operation=_read)

    def list_performance_snapshots(self, *, limit: int) -> dict[str, Any]:
        def _read() -> dict[str, Any]:
            columns = self._table_columns("performance_snapshots")
            if not columns:
                return {"snapshots": []}

            def _col_or_null(name: str) -> str:
                return name if name in columns else f"NULL AS {name}"

            select_cols = [
                _col_or_null("snapshot_time"),
                _col_or_null("model_version"),
                _col_or_null("hit_rate"),
                _col_or_null("total_predictions"),
                _col_or_null("sharpe_ratio"),
                _col_or_null("max_drawdown"),
            ]
            order_col = self._first_available(columns, ("snapshot_time", "id"))
            sql = (
                f"SELECT {', '.join(select_cols)} FROM performance_snapshots"  # noqa: S608
            )
            if order_col is not None:
                sql += f" ORDER BY {order_col} DESC"
            sql += " LIMIT :limit"

            with self.engine.connect() as connection:
                rows = [
                    dict(row)
                    for row in connection.execute(text(sql), {"limit": limit}).mappings().all()
                ]
            return {"snapshots": rows}

        return self._run_read(step="system.performance_snapshots", operation=_read)

    def get_system_activity_summary(self) -> dict[str, Any]:
        def _read() -> dict[str, Any]:
            latest_monitor_log = self._latest_table_value(
                table_name="system_logs",
                candidates=("timestamp", "created_at"),
                filters={"service": "monitoring_agent"},
            )
            if latest_monitor_log is None:
                latest_monitor_log = self._latest_table_value(
                    table_name="system_logs",
                    candidates=("timestamp", "created_at"),
                )

            return {
                "latest_snapshot": self._latest_table_value(
                    table_name="performance_snapshots",
                    candidates=("snapshot_time",),
                ),
                "latest_monitor_log": latest_monitor_log,
                "latest_retraining": self._latest_table_value(
                    table_name="retraining_events",
                    candidates=("started_at",),
                ),
            }

        return self._run_read(step="gui.system_status", operation=_read)

    def list_recent_system_events(self, *, limit: int) -> list[dict[str, Any]]:
        payload = self.list_system_logs(limit=limit)
        return [
            {
                "time": row["timestamp"],
                "level": row["level"],
                "message": row["message"],
            }
            for row in payload["logs"]
        ]

    def get_latest_prediction_payload(self) -> dict[str, Any] | None:
        def _read() -> dict[str, Any] | None:
            columns = self._table_columns("prediction_outcomes")
            if not columns:
                return None

            field_candidates: dict[str, tuple[str, ...]] = {
                "timestamp_utc": ("timestamp_utc", "prediction_timestamp"),
                "predicted_direction": ("predicted_direction",),
                "predicted_return": ("predicted_return",),
                "predicted_price": ("predicted_price",),
                "model_version": ("model_version",),
                "created_at": ("created_at", "prediction_timestamp", "timestamp_utc"),
                "p_up": ("p_up",),
                "p_down": ("p_down",),
                "confidence_raw": ("confidence",),
            }

            expressions: list[str] = []
            for alias, candidates in field_candidates.items():
                selected = self._first_available(columns, candidates)
                if selected is None:
                    expressions.append(f"NULL AS {alias}")
                else:
                    expressions.append(f"{selected} AS {alias}")

            order_column = self._first_available(
                columns,
                ("created_at", "timestamp_utc", "prediction_timestamp", "id"),
            )
            sql = (
                f"SELECT {', '.join(expressions)} FROM prediction_outcomes"  # noqa: S608
            )
            if order_column is not None:
                sql += f" ORDER BY {order_column} DESC"
            sql += " LIMIT 1"

            with self.engine.connect() as connection:
                row = connection.execute(text(sql)).mappings().first()
            if row is None:
                return None

            p_up = row["p_up"]
            p_down = row["p_down"]
            confidence = row["confidence_raw"]
            if confidence is None:
                candidates = [value for value in (p_up, p_down) if value is not None]
                confidence = max(candidates) if candidates else None

            return {
                "timestamp_utc": row["timestamp_utc"],
                "direction": row["predicted_direction"] or "flat",
                "predicted_return": (
                    row["predicted_return"] if row["predicted_return"] is not None else 0.0
                ),
                "predicted_price": row["predicted_price"],
                "model_version": row["model_version"],
                "created_at": row["created_at"] or row["timestamp_utc"],
                "p_up": p_up,
                "p_down": p_down,
                "confidence": confidence,
            }

        return self._run_read(step="gui.latest_prediction", operation=_read)

    def get_timeline_prediction_rows(
        self,
        *,
        freq: str,
        horizon: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        def _read() -> list[dict[str, Any]]:
            columns = self._table_columns("prediction_outcomes")
            if not columns:
                return []

            if "timestamp_utc" in columns:
                timestamp_expr = "timestamp_utc AS timestamp_utc"
            elif "prediction_timestamp" in columns:
                timestamp_expr = "prediction_timestamp AS timestamp_utc"
            else:
                return []

            if "freq" not in columns or "horizon" not in columns:
                return []

            select_exprs = [
                timestamp_expr,
                (
                    "predicted_direction"
                    if "predicted_direction" in columns
                    else "'flat' AS predicted_direction"
                ),
                "p_up" if "p_up" in columns else "NULL AS p_up",
                "p_down" if "p_down" in columns else "NULL AS p_down",
                "predicted_return" if "predicted_return" in columns else "NULL AS predicted_return",
                "predicted_price" if "predicted_price" in columns else "NULL AS predicted_price",
                "actual_return" if "actual_return" in columns else "NULL AS actual_return",
                "actual_direction" if "actual_direction" in columns else "NULL AS actual_direction",
                "correct" if "correct" in columns else "NULL AS correct",
            ]

            order_clause = "ORDER BY timestamp_utc DESC"
            if "created_at" in columns:
                order_clause = "ORDER BY timestamp_utc DESC, created_at DESC"
            elif "id" in columns:
                order_clause = "ORDER BY timestamp_utc DESC, id DESC"

            sql = (
                f"SELECT {', '.join(select_exprs)} FROM prediction_outcomes "  # noqa: S608
                f"WHERE freq = :freq AND horizon = :horizon {order_clause} LIMIT :limit"
            )
            with self.engine.connect() as connection:
                return [
                    dict(row)
                    for row in connection.execute(
                        text(sql),
                        {"freq": freq, "horizon": horizon, "limit": limit},
                    ).mappings().all()
                ]

        return self._run_read(step="gui.timeline", operation=_read)

    def list_prediction_pairs(
        self,
        *,
        default_freq: str,
        default_horizon: str,
    ) -> tuple[list[str], list[str]]:
        def _read() -> tuple[list[str], list[str]]:
            freqs = {default_freq}
            horizons = {default_horizon}
            columns = self._table_columns("prediction_outcomes")
            if {"freq", "horizon"}.issubset(columns):
                sql = (
                    "SELECT DISTINCT freq, horizon FROM prediction_outcomes "
                    "WHERE freq IS NOT NULL AND horizon IS NOT NULL"
                )
                with self.engine.connect() as connection:
                    rows = connection.execute(text(sql)).all()
                for freq, horizon in rows:
                    freqs.add(str(freq))
                    horizons.add(str(horizon))

            return (
                sorted(freqs, key=_duration_sort_key),
                sorted(horizons, key=_duration_sort_key),
            )

        return self._run_read(step="gui.timeline_filters", operation=_read)

    def store_prediction(
        self,
        session: Session,
        timestamp_utc: datetime,
        predicted_direction: str,
        model_version: str,
        freq: str,
        horizon: str,
        predicted_return: float | None = None,
        predicted_price: float | None = None,
        p_up: float = 0.0,
        p_down: float = 0.0,
        p_flat: float = 0.0,
        features_used: dict[str, Any] | None = None,
    ) -> PredictionOutcome:
        """Insert a new prediction row."""
        prediction = PredictionOutcome(
            timestamp_utc=timestamp_utc,
            prediction_timestamp=_utcnow(),
            predicted_direction=predicted_direction,
            p_up=p_up,
            p_down=p_down,
            p_flat=p_flat,
            predicted_return=predicted_return,
            predicted_price=predicted_price,
            model_version=model_version,
            freq=freq,
            horizon=horizon,
            features_used=features_used,
        )
        session.add(prediction)
        session.flush()
        return prediction

    def get_unrealized_predictions(
        self,
        session: Session,
        freq: str,
        horizon: str,
        cutoff_time: datetime | None = None,
    ) -> list[PredictionOutcome]:
        """Return predictions without realized returns for the requested config."""
        query = session.query(PredictionOutcome).filter(
            PredictionOutcome.actual_return.is_(None),
            PredictionOutcome.freq == freq,
            PredictionOutcome.horizon == horizon,
        )
        if cutoff_time is not None:
            query = query.filter(PredictionOutcome.timestamp_utc < cutoff_time)
        return list(query.order_by(PredictionOutcome.timestamp_utc.asc()).all())

    def get_prediction_counts(
        self, session: Session, freq: str, horizon: str
    ) -> dict[str, int | str]:  # noqa: E501
        """Return pair-scoped total/unrealized/realized prediction counts."""
        base_query = session.query(PredictionOutcome).filter(
            PredictionOutcome.freq == freq,
            PredictionOutcome.horizon == horizon,
        )
        total_predictions = int(base_query.count())
        realized_predictions = int(
            base_query.filter(PredictionOutcome.actual_return.is_not(None)).count()
        )
        unrealized_predictions = total_predictions - realized_predictions
        return {
            "freq": freq,
            "horizon": horizon,
            "total_predictions": total_predictions,
            "unrealized_predictions": unrealized_predictions,
            "realized_predictions": realized_predictions,
        }

    def realize_prediction(
        self,
        session: Session,
        prediction_id: int,
        actual_return: float,
        actual_direction: str,
    ) -> PredictionOutcome:
        """Fill realized fields for an existing prediction row."""
        prediction = session.get(PredictionOutcome, prediction_id)
        if prediction is None:
            raise ValueError(f"Prediction {prediction_id} not found.")

        prediction.actual_return = actual_return
        prediction.actual_direction = actual_direction
        prediction.correct = prediction.predicted_direction == actual_direction
        prediction.realized_at = _utcnow()
        session.flush()
        return prediction

    def get_recent_predictions(
        self,
        session: Session,
        freq: str,
        horizon: str,
        days: int = 30,
        realized_only: bool = True,
    ) -> list[PredictionOutcome]:
        """Return recent predictions for performance calculations."""
        cutoff = _utcnow() - timedelta(days=days)
        query = session.query(PredictionOutcome).filter(
            PredictionOutcome.freq == freq,
            PredictionOutcome.horizon == horizon,
            PredictionOutcome.created_at >= cutoff,
        )
        if realized_only:
            query = query.filter(PredictionOutcome.actual_return.is_not(None))
        return list(query.order_by(desc(PredictionOutcome.timestamp_utc)).all())

    def store_model_version(
        self,
        session: Session,
        version: str,
        freq: str,
        horizon: str,
        training_start: datetime,
        training_end: datetime,
        training_samples: int,
        cv_score: float | None,
        features: list[str] | None,
        hyperparameters: dict[str, Any] | None,
        training_metadata: dict[str, Any] | None,
        is_active: bool = True,
    ) -> ModelVersion:
        """Insert model metadata for a newly trained model."""
        model = ModelVersion(
            version=version,
            freq=freq,
            horizon=horizon,
            training_start=training_start,
            training_end=training_end,
            training_samples=training_samples,
            cv_score=cv_score,
            features=features,
            hyperparameters=hyperparameters,
            is_active=is_active,
            training_metadata=training_metadata,
        )
        session.add(model)
        session.flush()
        return model

    def get_active_model(self, session: Session, freq: str, horizon: str) -> ModelVersion | None:
        """Fetch the active model for a frequency and horizon pair."""
        return (
            session.query(ModelVersion)
            .filter(
                ModelVersion.freq == freq,
                ModelVersion.horizon == horizon,
                ModelVersion.is_active.is_(True),
            )
            .first()
        )

    def deactivate_old_models(self, session: Session, freq: str, horizon: str) -> int:
        """Deactivate all models for the requested frequency/horizon pair."""
        updated_count = (
            session.query(ModelVersion)
            .filter(ModelVersion.freq == freq, ModelVersion.horizon == horizon)
            .update(
                {
                    "is_active": False,
                    "replaced_at": _utcnow(),
                },
                synchronize_session=False,
            )
        )
        session.flush()
        return int(updated_count or 0)

    def create_retraining_event(
        self,
        session: Session,
        trigger_reason: str,
        trigger_metrics: dict[str, Any] | None,
        old_model_version: str | None = None,
    ) -> RetrainingEvent:
        """Create a retraining event row with status `started`."""
        event = RetrainingEvent(
            trigger_reason=trigger_reason,
            trigger_metrics=trigger_metrics,
            old_model_version=old_model_version,
            status="started",
            started_at=_utcnow(),
        )
        session.add(event)
        session.flush()
        return event

    def complete_retraining_event(
        self,
        session: Session,
        event_id: int,
        new_model_version: str,
        cv_improvement: float,
        training_duration_seconds: float,
    ) -> RetrainingEvent:
        """Update an event row with success metadata."""
        event = session.get(RetrainingEvent, event_id)
        if event is None:
            raise ValueError(f"Retraining event {event_id} not found.")
        event.status = "completed"
        event.new_model_version = new_model_version
        event.cv_improvement = cv_improvement
        event.training_duration_seconds = training_duration_seconds
        event.completed_at = _utcnow()
        session.flush()
        return event

    def fail_retraining_event(
        self,
        session: Session,
        event_id: int,
        error_message: str,
    ) -> RetrainingEvent:
        """Update an event row with failure metadata."""
        event = session.get(RetrainingEvent, event_id)
        if event is None:
            raise ValueError(f"Retraining event {event_id} not found.")
        event.status = "failed"
        event.error_message = error_message
        event.completed_at = _utcnow()
        session.flush()
        return event

    def finalize_retraining_success(
        self,
        *,
        event_id: int,
        new_model_version: str,
        freq: str,
        horizon: str,
        cv_improvement: float,
        training_duration_seconds: float,
    ) -> None:
        with self.session() as session:
            self.deactivate_old_models(session, freq, horizon)
            candidate = (
                session.query(ModelVersion)
                .filter(
                    ModelVersion.version == new_model_version,
                    ModelVersion.freq == freq,
                    ModelVersion.horizon == horizon,
                )
                .first()
            )
            if candidate is None:
                raise ValueError(f"Model version not found for deployment: {new_model_version}")
            candidate.is_active = True
            candidate.deployed_at = _utcnow()
            self.complete_retraining_event(
                session=session,
                event_id=event_id,
                new_model_version=new_model_version,
                cv_improvement=cv_improvement,
                training_duration_seconds=training_duration_seconds,
            )

    def finalize_retraining_failure(
        self,
        *,
        event_id: int,
        error_message: str,
    ) -> None:
        with self.session() as session:
            self.fail_retraining_event(
                session=session,
                event_id=event_id,
                error_message=error_message,
            )

    def store_performance_snapshot(
        self,
        session: Session,
        model_version: str,
        freq: str,
        horizon: str,
        window_days: int,
        metrics: dict[str, Any],
    ) -> PerformanceSnapshot:
        """Insert a performance snapshot row from a metrics dictionary."""
        snapshot = PerformanceSnapshot(
            model_version=model_version,
            freq=freq,
            horizon=horizon,
            snapshot_time=_utcnow(),
            window_days=window_days,
            total_predictions=int(metrics.get("total_predictions", 0)),
            realized_predictions=int(metrics.get("realized_predictions", 0)),
            hit_rate=metrics.get("hit_rate"),
            sharpe_ratio=metrics.get("sharpe_ratio"),
            avg_return=metrics.get("avg_return"),
            max_drawdown=metrics.get("max_drawdown"),
            win_streak=metrics.get("win_streak"),
            lose_streak=metrics.get("lose_streak"),
            calibration_score=metrics.get("calibration_score"),
        )
        session.add(snapshot)
        session.flush()
        return snapshot

    def log(
        self,
        session: Session,
        level: str,
        service: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> SystemLog:
        """Insert a system log row."""
        log_entry = SystemLog(
            level=level,
            service=service,
            message=message,
            details=details,
        )
        session.add(log_entry)
        session.flush()
        return log_entry
