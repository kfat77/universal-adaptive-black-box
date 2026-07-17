"""Validation split construction for numerical surrogate experiments."""

from typing import Iterable

import numpy as np
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    LeaveOneGroupOut,
    RepeatedKFold,
    TimeSeriesSplit,
    train_test_split,
)


def build_splits(
    strategy: str,
    n_samples: int,
    n_splits: int,
    random_state: int,
    groups: np.ndarray | None = None,
    holdout_fraction: float = 0.2,
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    """Return deterministic train/validation indices for a supported strategy."""
    indices = np.arange(n_samples)
    if strategy in {"kfold", "repeated_kfold"}:
        if not 2 <= n_splits <= n_samples // 2:
            raise ValueError("n_splits must leave at least two samples in every validation fold.")
    if strategy == "kfold":
        return KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(indices)
    if strategy == "repeated_kfold":
        return RepeatedKFold(n_splits=n_splits, n_repeats=2, random_state=random_state).split(
            indices
        )
    if strategy in {"group_kfold", "leave_one_group_out"}:
        if groups is None or len(groups) != n_samples:
            raise ValueError(f"{strategy} requires one group label per sample.")
        return (
            GroupKFold(n_splits=n_splits).split(indices, groups=groups)
            if strategy == "group_kfold"
            else LeaveOneGroupOut().split(indices, groups=groups)
        )
    if strategy == "time_series":
        if not 2 <= n_splits < n_samples:
            raise ValueError("time_series requires 2 <= n_splits < sample count.")
        return TimeSeriesSplit(n_splits=n_splits).split(indices)
    if strategy == "holdout":
        if not 0 < holdout_fraction < 0.5:
            raise ValueError("holdout_fraction must be between 0 and 0.5.")
        train, validation = train_test_split(
            indices, test_size=holdout_fraction, random_state=random_state
        )
        return [(train, validation)]
    raise ValueError(
        "validation_strategy must be kfold, repeated_kfold, group_kfold, leave_one_group_out, time_series, or holdout."
    )
