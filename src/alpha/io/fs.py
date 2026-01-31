"""Local filesystem helpers for parquet storage."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api.types import CategoricalDtype


def _normalize_categoricals(frame: pd.DataFrame) -> pd.DataFrame:
    """Cast categorical columns back to their category dtype."""
    for column, dtype in frame.dtypes.items():
        if isinstance(dtype, CategoricalDtype):
            category_dtype = frame[column].cat.categories.dtype
            frame[column] = frame[column].astype(category_dtype)
    return frame


def read_parquet(path: str | Path, filters: Any | None = None, **kwargs: Any) -> pd.DataFrame:
    """Read a parquet file from disk into a DataFrame."""
    file_path = Path(path).expanduser().resolve()
    read_kwargs: dict[str, Any] = {"engine": "pyarrow"}
    if filters is not None:
        read_kwargs["filters"] = filters
    read_kwargs.update(kwargs)
    result = pd.read_parquet(file_path, **read_kwargs)
    return _normalize_categoricals(result)


def write_parquet(
    data: pd.DataFrame,
    path: str | Path,
    *,
    partition_cols: Sequence[str] | None = None,
    **kwargs: Any,
) -> None:
    """Persist a DataFrame to parquet storage."""
    target_path = Path(path).expanduser().resolve()

    if partition_cols:
        target_path.mkdir(parents=True, exist_ok=True)
    else:
        target_path.parent.mkdir(parents=True, exist_ok=True)

    write_kwargs: dict[str, Any] = {"engine": "pyarrow", "index": False}
    write_kwargs.update(kwargs)
    data.to_parquet(
        target_path,
        partition_cols=list(partition_cols) if partition_cols else None,
        **write_kwargs,
    )
