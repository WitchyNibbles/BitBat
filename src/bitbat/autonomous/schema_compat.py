"""Compatibility contract and schema inspection helpers for autonomous DB tables."""

from __future__ import annotations

from dataclasses import dataclass

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


RUNTIME_SCHEMA_CONTRACT: dict[str, dict[str, ColumnContract]] = {
    "prediction_outcomes": PREDICTION_OUTCOMES_CONTRACT,
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


def _transaction(connection: Connection):
    if connection.in_transaction():
        return connection.begin_nested()
    return connection.begin()


def upgrade_schema_compatibility(
    database_url: str | None = None,
    *,
    engine: Engine | Connection | None = None,
) -> SchemaUpgradeResult:
    """Apply idempotent additive upgrades for missing runtime columns."""
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
        with _transaction(connection):
            for table in report_before.tables:
                for column_name in table.addable_missing_columns:
                    contract = RUNTIME_SCHEMA_CONTRACT[table.table_name][column_name]
                    connection.execute(text(
                        f'ALTER TABLE "{table.table_name}" '
                        f'ADD COLUMN "{column_name}" {contract.sql_type}'
                    ))
                    actions.append(SchemaUpgradeAction(
                        table_name=table.table_name,
                        column_name=column_name,
                        sql_type=contract.sql_type,
                    ))
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
