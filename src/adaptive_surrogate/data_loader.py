"""Loading and validating numerical tabular datasets for black-box training."""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class TabularDataset:
    """Validated model arrays plus the source column names used to build them."""

    X: np.ndarray
    Y: np.ndarray
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]


def _column_names(columns: str | Sequence[str], name: str) -> tuple[str, ...]:
    names = (columns,) if isinstance(columns, str) else tuple(columns)
    if not names or len(names) != len(set(names)):
        raise ValueError(f"{name} must contain one or more unique column names.")
    return names


def load_tabular_data(
    path: str | Path,
    target_columns: str | Sequence[str],
    feature_columns: str | Sequence[str] | None = None,
    sheet_name: str | int = 0,
) -> TabularDataset:
    """Load finite numerical feature and target columns from CSV or Excel.

    When ``feature_columns`` is omitted, every column except the selected targets
    becomes an input feature. Excel files use ``sheet_name``; it is ignored for CSV.
    """
    path = Path(path)
    import pandas as pd

    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix.lower() in {".xls", ".xlsx"}:
        frame = pd.read_excel(path, sheet_name=sheet_name)
    else:
        raise ValueError("Only .csv, .xls, and .xlsx files are supported.")

    if frame.columns.duplicated().any():
        raise ValueError("Source column names must be unique.")
    if not all(isinstance(column, str) for column in frame.columns):
        raise ValueError("Source column names must be strings.")

    targets = _column_names(target_columns, "target_columns")
    features = (
        _column_names(feature_columns, "feature_columns")
        if feature_columns is not None
        else tuple(column for column in frame.columns if column not in targets)
    )
    selected = features + targets
    missing = [column for column in selected if column not in frame.columns]
    if missing:
        raise ValueError(f"Selected columns are missing from the file: {missing}")
    if not features:
        raise ValueError("At least one feature column is required.")
    if set(features) & set(targets):
        raise ValueError("Feature and target columns must not overlap.")

    selected_frame = frame.loc[:, list(selected)]
    if any(not pd.api.types.is_numeric_dtype(selected_frame[column]) for column in selected):
        raise ValueError("All selected feature and target columns must be numeric.")
    values = selected_frame.to_numpy(dtype=np.float64)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Selected columns must contain at least one row of finite values.")
    split_index = len(features)
    return TabularDataset(values[:, :split_index], values[:, split_index:], features, targets)
