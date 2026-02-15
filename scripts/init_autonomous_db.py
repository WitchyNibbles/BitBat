"""Initialize the autonomous monitoring database schema."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sqlalchemy import inspect

# Allow direct script execution without package installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bitbat.autonomous.models import Base, create_database_engine, init_database

EXPECTED_TABLES = [
    "prediction_outcomes",
    "model_versions",
    "retraining_events",
    "performance_snapshots",
    "system_logs",
]


def check_existing_tables(database_url: str) -> list[str]:
    """Return expected autonomous tables that already exist."""
    inspector = inspect(create_database_engine(database_url))
    existing = set(inspector.get_table_names())
    return sorted(table for table in EXPECTED_TABLES if table in existing)


def _ensure_sqlite_parent(database_url: str) -> None:
    if not database_url.startswith("sqlite:///"):
        return
    sqlite_path = database_url.replace("sqlite:///", "", 1)
    if sqlite_path == ":memory:":
        return
    db_path = Path(sqlite_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize autonomous database.")
    parser.add_argument(
        "--database-url",
        default="sqlite:///data/autonomous.db",
        help="SQLAlchemy connection URL.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Drop and recreate existing autonomous tables.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _ensure_sqlite_parent(args.database_url)

    engine = create_database_engine(args.database_url)
    existing = check_existing_tables(args.database_url)

    if existing and not args.force:
        print(f"Found existing autonomous tables: {existing}")
        print("Rerun with --force to drop and recreate them.")
        return 1

    if args.force and existing:
        print(f"Dropping existing tables: {existing}")
        Base.metadata.drop_all(engine)

    print(f"Creating autonomous schema at {args.database_url}")
    init_database(args.database_url, engine=engine)

    created = check_existing_tables(args.database_url)
    print(f"Created tables: {created}")

    if len(created) != len(EXPECTED_TABLES):
        print(
            f"Expected {len(EXPECTED_TABLES)} tables but found {len(created)} after init.",
        )
        return 1

    print("Autonomous database initialization complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
