"""DuckDB helper functions."""

from __future__ import annotations

from typing import Any

import duckdb
import pandas as pd


def query(sql: str, **params: Any) -> pd.DataFrame:
    """Execute a SQL query against DuckDB and return the results as a DataFrame."""
    connection = duckdb.connect(database=":memory:")
    try:
        execution = connection.execute(sql, params) if params else connection.execute(sql)
        result = execution.df()
    finally:
        connection.close()
    return result
