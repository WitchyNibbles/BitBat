"""Compatibility contract and schema inspection helpers for autonomous DB tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sqlalchemy import inspect, text
from sqlalchemy.engine import Connection, Engine


@dataclass(frozen=True, slots=True)
class ColumnContract:
    """Describes one required column in the runtime schema contract."""

    sql_type: str
    nullable: bool = True
    additive: bool = True


@dataclass(frozen=True, slots=True)
class TableSchemaAudit:
    """Compatibility audit details for a single table."""

    table_name: str
    required_columns: tuple[str, ...]
    existing_columns: tuple[str, ...]
    missing_columns: tuple[str, ...]
    addable_missing_columns: tuple[str, ...]
    blocking_missing_columns: tuple[str, ...]

    @property
    def is_compatible(self) -> bool:
        return not self.missing_columns


@dataclass(frozen=True, slots=True)
class SchemaAuditReport:
    """Compatibility audit result for all runtime-required tables."""

    tables: tuple[TableSchemaAudit, ...]

    @property
    def is_compatible(self) -> bool:
        return all(table.is_compatible for table in self.tables)

    @property
    def missing_columns(self) -> dict[str, tuple[str, ...]]:
        return {
            table.table_name: table.missing_columns
            for table in self.tables
            if table.missing_columns
        }

    @property
    def can_auto_upgrade(self) -> bool:
        return all(not table.blocking_missing_columns for table in self.tables)

    @property
    def missing_column_count(self) -> int:
        return sum(len(table.missing_columns) for table in self.tables)


@dataclass(frozen=True, slots=True)
class SchemaUpgradeAction:
    """One additive migration action applied during compatibility upgrade."""

    table_name: str
    column_name: str
    sql_type: str


@dataclass(frozen=True, slots=True)
class SchemaUpgradeResult:
    """Result of running additive schema upgrade."""

    report_before: SchemaAuditReport
    report_after: SchemaAuditReport
    actions: tuple[SchemaUpgradeAction, ...]

    @property
    def upgraded(self) -> bool:
        return bool(self.actions)

    @property
    def is_compatible(self) -> bool:
        return self.report_after.is_compatible

    @property
    def upgrade_state(self) -> Literal["upgraded", "already_compatible", "incompatible"]:
        if self.actions:
            return "upgraded"
        if self.report_after.is_compatible:
            return "already_compatible"
        return "incompatible"

    @property
    def operation_count(self) -> int:
        return len(self.actions)

    @property
    def missing_columns_before(self) -> int:
        return self.report_before.missing_column_count

    @property
    def missing_columns_after(self) -> int:
        return self.report_after.missing_column_count

    @property
    def status(self) -> dict[str, str | int]:
        return {
            "upgrade_state": self.upgrade_state,
            "operations_applied": self.operation_count,
            "missing_columns_before": self.missing_columns_before,
            "missing_columns_after": self.missing_columns_after,
        }


class SchemaCompatibilityError(RuntimeError):
    """Raised when runtime schema requirements are not satisfied."""

    def __init__(self, report: SchemaAuditReport, database_url: str | None = None) -> None:
        self.report = report
        self.database_url = database_url
        super().__init__(self._build_message())

    def _build_message(self) -> str:
        missing = []
        for table_name, columns in self.report.missing_columns.items():
            missing.append(f"{table_name}({', '.join(columns)})")
        missing_text = "; ".join(missing) if missing else "unknown compatibility issue"
        db_target = self.database_url or "sqlite:///data/autonomous.db"
        lines = [
            f"Autonomous schema compatibility check failed: {missing_text}.",
            (
                "Run `python scripts/init_autonomous_db.py --database-url "
                f"{db_target} --audit` to inspect current state."
            ),
        ]
        if self.report.can_auto_upgrade:
            lines.append(
                "Run `python scripts/init_autonomous_db.py --database-url "
                f"{db_target} --upgrade` to apply additive compatibility upgrades."
            )
        else:
            lines.append(
                "Blocking non-additive columns are missing; use `--force` only if table "
                "recreation is acceptable."
            )
        return " ".join(lines)


PREDICTION_OUTCOMES_CONTRACT: dict[str, ColumnContract] = {
    # Non-additive required columns are expected to exist once table creation succeeds.
    "id": ColumnContract("INTEGER", nullable=False, additive=False),
    "timestamp_utc": ColumnContract("DATETIME", nullable=False, additive=False),
    "prediction_timestamp": ColumnContract("DATETIME", nullable=False, additive=False),
    "predicted_direction": ColumnContract("VARCHAR(10)", nullable=False, additive=False),
    "model_version": ColumnContract("VARCHAR(64)", nullable=False, additive=False),
    "freq": ColumnContract("VARCHAR(16)", nullable=False, additive=False),
    "horizon": ColumnContract("VARCHAR(16)", nullable=False, additive=False),
    "created_at": ColumnContract("DATETIME", nullable=False, additive=False),
    # Additive/runtime compatibility columns for legacy DBs.
    "p_up": ColumnContract("FLOAT", nullable=True, additive=True),
    "p_down": ColumnContract("FLOAT", nullable=True, additive=True),
    "p_flat": ColumnContract("FLOAT", nullable=True, additive=True),
    "predicted_return": ColumnContract("FLOAT", nullable=True, additive=True),
    "predicted_price": ColumnContract("FLOAT", nullable=True, additive=True),
    "actual_return": ColumnContract("FLOAT", nullable=True, additive=True),
    "actual_direction": ColumnContract("VARCHAR(10)", nullable=True, additive=True),
    "correct": ColumnContract("BOOLEAN", nullable=True, additive=True),
    "features_used": ColumnContract("JSON", nullable=True, additive=True),
    "realized_at": ColumnContract("DATETIME", nullable=True, additive=True),
}


PERFORMANCE_SNAPSHOTS_CONTRACT: dict[str, ColumnContract] = {
    # Non-additive required columns are expected to exist once table creation succeeds.
    "id": ColumnContract("INTEGER", nullable=False, additive=False),
    "model_version": ColumnContract("VARCHAR(64)", nullable=False, additive=False),
    "freq": ColumnContract("VARCHAR(16)", nullable=False, additive=False),
    "horizon": ColumnContract("VARCHAR(16)", nullable=False, additive=False),
    "snapshot_time": ColumnContract("DATETIME", nullable=False, additive=False),
    "window_days": ColumnContract("INTEGER", nullable=False, additive=False),
    "total_predictions": ColumnContract("INTEGER", nullable=False, additive=False),
    "realized_predictions": ColumnContract("INTEGER", nullable=False, additive=False),
    "created_at": ColumnContract("DATETIME", nullable=False, additive=False),
    # Additive/runtime compatibility columns for legacy DBs.
    "hit_rate": ColumnContract("FLOAT", nullable=True, additive=True),
    "sharpe_ratio": ColumnContract("FLOAT", nullable=True, additive=True),
    "avg_return": ColumnContract("FLOAT", nullable=True, additive=True),
    "max_drawdown": ColumnContract("FLOAT", nullable=True, additive=True),
    "win_streak": ColumnContract("INTEGER", nullable=True, additive=True),
    "lose_streak": ColumnContract("INTEGER", nullable=True, additive=True),
    "calibration_score": ColumnContract("FLOAT", nullable=True, additive=True),
    "mae": ColumnContract("FLOAT", nullable=True, additive=True),
    "rmse": ColumnContract("FLOAT", nullable=True, additive=True),
    "directional_accuracy": ColumnContract("FLOAT", nullable=True, additive=True),
}


RUNTIME_SCHEMA_CONTRACT: dict[str, dict[str, ColumnContract]] = {
    "prediction_outcomes": PREDICTION_OUTCOMES_CONTRACT,
    "performance_snapshots": PERFORMANCE_SNAPSHOTS_CONTRACT,
}


# ---------------------------------------------------------------------------
# CHECK constraint evolution contract
# ---------------------------------------------------------------------------
# Maps table_name -> constraint_name -> set of values that MUST be present in
# the CHECK expression.  Used to detect stale CHECK constraints on existing
# tables (SQLite cannot ALTER a CHECK; the table must be recreated).
# ---------------------------------------------------------------------------
CHECK_CONSTRAINT_CONTRACT: dict[str, dict[str, set[str]]] = {
    "retraining_events": {
        "ck_trigger_reason": {
            "drift_detected",
            "scheduled",
            "manual",
            "poor_performance",
            "continuous",
        },
    },
}


def required_columns_for_table(table_name: str) -> tuple[str, ...]:
    """Return required columns for a runtime table, sorted for deterministic reporting."""
    contract = RUNTIME_SCHEMA_CONTRACT.get(table_name, {})
    return tuple(sorted(contract.keys()))


def _resolve_engine(
    database_url: str | None,
    engine: Engine | Connection | None,
) -> tuple[Engine | Connection, bool]:
    if engine is not None:
        return engine, False
    from .models import create_database_engine

    resolved_url = database_url or "sqlite:///data/autonomous.db"
    return create_database_engine(resolved_url), True


def _get_columns(
    connection: Connection,
    table_name: str,
) -> tuple[str, ...]:
    inspector = inspect(connection)
    table_names = set(inspector.get_table_names())
    if table_name not in table_names:
        return ()
    return tuple(sorted(column["name"] for column in inspector.get_columns(table_name)))


def _audit_table(connection: Connection, table_name: str) -> TableSchemaAudit:
    contract = RUNTIME_SCHEMA_CONTRACT[table_name]
    required = tuple(sorted(contract.keys()))
    existing = _get_columns(connection, table_name)
    missing = tuple(column for column in required if column not in set(existing))
    addable = tuple(column for column in missing if contract[column].additive)
    blocking = tuple(column for column in missing if not contract[column].additive)
    return TableSchemaAudit(
        table_name=table_name,
        required_columns=required,
        existing_columns=existing,
        missing_columns=missing,
        addable_missing_columns=addable,
        blocking_missing_columns=blocking,
    )


def audit_schema_compatibility(
    database_url: str | None = None,
    *,
    engine: Engine | Connection | None = None,
) -> SchemaAuditReport:
    """Inspect runtime tables and report missing required columns."""
    target, owns_engine = _resolve_engine(database_url, engine)
    created_connection = False
    if isinstance(target, Connection):
        connection = target
    else:
        connection = target.connect()
        created_connection = True

    try:
        tables = tuple(
            _audit_table(connection, table_name)
            for table_name in sorted(RUNTIME_SCHEMA_CONTRACT.keys())
        )
        return SchemaAuditReport(tables=tables)
    finally:
        if created_connection:
            connection.close()
        if owns_engine and isinstance(target, Engine):
            target.dispose()


def format_schema_audit(report: SchemaAuditReport) -> str:
    """Render a deterministic, operator-friendly compatibility report."""
    lines = ["Autonomous schema compatibility audit"]
    for table in report.tables:
        status = "compatible" if table.is_compatible else "incompatible"
        lines.append(f"- {table.table_name}: {status}")
        if table.missing_columns:
            lines.append(f"  missing: {', '.join(table.missing_columns)}")
    return "\n".join(lines)


def format_missing_columns(report: SchemaAuditReport) -> str:
    """Return compact `table(col1, col2)` text for user-facing error messages."""
    pairs = []
    for table_name, columns in report.missing_columns.items():
        pairs.append(f"{table_name}({', '.join(columns)})")
    return "; ".join(pairs)


def _get_check_constraint_sql(connection: Connection, table_name: str) -> dict[str, str]:
    """Return {constraint_name: sql_expression} for CHECK constraints on *table_name*.

    Works by parsing the CREATE TABLE DDL from sqlite_master since SQLAlchemy's
    inspector does not reliably expose CHECK constraint expressions for SQLite.
    """
    import re

    result = connection.execute(
        text("SELECT sql FROM sqlite_master WHERE type='table' AND name=:t"),
        {"t": table_name},
    )
    row = result.fetchone()
    if row is None or row[0] is None:
        return {}

    ddl: str = row[0]
    # Match CONSTRAINT <name> CHECK (<expression>)
    pattern = r"CONSTRAINT\s+(\w+)\s+CHECK\s*\((.+?)\)(?:\s*,|\s*\))"
    checks: dict[str, str] = {}
    for match in re.finditer(pattern, ddl, re.IGNORECASE | re.DOTALL):
        checks[match.group(1)] = match.group(2).strip()
    return checks


def _check_constraints_are_current(connection: Connection) -> list[tuple[str, str]]:
    """Return list of (table_name, constraint_name) pairs with stale CHECK constraints.

    A constraint is 'stale' when the contract requires values that are not
    present in the existing CHECK expression stored in the DB schema.
    """
    stale: list[tuple[str, str]] = []
    for table_name, constraints in CHECK_CONSTRAINT_CONTRACT.items():
        existing = _get_check_constraint_sql(connection, table_name)
        for constraint_name, required_values in constraints.items():
            sql_expr = existing.get(constraint_name, "")
            for value in required_values:
                if f"'{value}'" not in sql_expr:
                    stale.append((table_name, constraint_name))
                    break
    return stale


def _rebuild_table_with_current_schema(connection: Connection, table_name: str) -> None:
    """Recreate *table_name* using the current ORM schema, preserving existing data.

    Uses the standard SQLite table-rebuild pattern:
      1. Rename existing table to a temp name
      2. Create new table from ORM DDL (with updated constraints)
      3. Copy data from temp table, using only columns that exist in both
      4. Drop temp table
    """
    from sqlalchemy.schema import CreateTable

    from .models import Base

    orm_table = Base.metadata.tables.get(table_name)
    if orm_table is None:
        return

    inspector = inspect(connection)
    table_names = set(inspector.get_table_names())
    if table_name not in table_names:
        return

    old_columns = {col["name"] for col in inspector.get_columns(table_name)}
    new_columns = {col.name for col in orm_table.columns}
    shared_columns = sorted(old_columns & new_columns)

    if not shared_columns:
        return

    temp_name = f"_upgrade_backup_{table_name}"
    cols_csv = ", ".join(f'"{c}"' for c in shared_columns)

    # Generate DDL from ORM metadata (includes updated CHECK constraints)
    create_ddl = (
        CreateTable(orm_table)
        .compile(
            dialect=connection.engine.dialect,
        )
        .string.strip()
    )

    connection.execute(text(f'ALTER TABLE "{table_name}" RENAME TO "{temp_name}"'))
    connection.execute(text(create_ddl))
    copy_sql = f'INSERT INTO "{table_name}" ({cols_csv}) SELECT {cols_csv} FROM "{temp_name}"'  # noqa: S608
    connection.execute(text(copy_sql))
    connection.execute(text(f'DROP TABLE "{temp_name}"'))


def _transaction(connection: Connection):
    if connection.in_transaction():
        return connection.begin_nested()
    return connection.begin()


def upgrade_schema_compatibility(
    database_url: str | None = None,
    *,
    engine: Engine | Connection | None = None,
) -> SchemaUpgradeResult:
    """Apply idempotent additive upgrades for missing columns and stale CHECK constraints."""
    target, owns_engine = _resolve_engine(database_url, engine)
    created_connection = False
    if isinstance(target, Connection):
        connection = target
    else:
        connection = target.connect()
        created_connection = True

    try:
        report_before = audit_schema_compatibility(engine=connection)
        actions: list[SchemaUpgradeAction] = []

        # Determine which tables actually exist so we skip ALTER on absent ones.
        upgrade_inspector = inspect(connection)
        existing_tables = set(upgrade_inspector.get_table_names())

        with _transaction(connection):
            for table in report_before.tables:
                if table.table_name not in existing_tables:
                    continue
                for column_name in table.addable_missing_columns:
                    contract = RUNTIME_SCHEMA_CONTRACT[table.table_name][column_name]
                    connection.execute(
                        text(
                            f'ALTER TABLE "{table.table_name}" '
                            f'ADD COLUMN "{column_name}" {contract.sql_type}'
                        )
                    )
                    actions.append(
                        SchemaUpgradeAction(
                            table_name=table.table_name,
                            column_name=column_name,
                            sql_type=contract.sql_type,
                        )
                    )

            # Rebuild tables whose CHECK constraints are stale (e.g. new enum
            # values added to trigger_reason).  SQLite does not support ALTER
            # CHECK, so the table must be recreated with data migration.
            stale_checks = _check_constraints_are_current(connection)
            rebuilt_tables: set[str] = set()
            for table_name, constraint_name in stale_checks:
                if table_name not in rebuilt_tables:
                    _rebuild_table_with_current_schema(connection, table_name)
                    rebuilt_tables.add(table_name)
                    actions.append(
                        SchemaUpgradeAction(
                            table_name=table_name,
                            column_name=f"__check__{constraint_name}",
                            sql_type="REBUILT",
                        )
                    )

        report_after = audit_schema_compatibility(engine=connection)
        return SchemaUpgradeResult(
            report_before=report_before,
            report_after=report_after,
            actions=tuple(actions),
        )
    finally:
        if created_connection:
            connection.close()
        if owns_engine and isinstance(target, Engine):
            target.dispose()


def ensure_schema_compatibility(
    database_url: str | None = None,
    *,
    engine: Engine | Connection | None = None,
    auto_upgrade: bool = False,
    raise_on_error: bool = True,
) -> SchemaAuditReport:
    """Audit (and optionally upgrade) runtime schema compatibility."""
    if auto_upgrade:
        result = upgrade_schema_compatibility(database_url=database_url, engine=engine)
        report = result.report_after
    else:
        report = audit_schema_compatibility(database_url=database_url, engine=engine)

    if raise_on_error and not report.is_compatible:
        raise SchemaCompatibilityError(report=report, database_url=database_url)
    return report
