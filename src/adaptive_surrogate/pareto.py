"""Dependency-free Pareto filtering for predicted multi-objective designs."""

import numpy as np


def non_dominated_mask(values: np.ndarray, directions: list[str]) -> np.ndarray:
    """Return a non-dominated mask for minimize/maximize objective directions."""
    points = np.asarray(values, dtype=float)
    if points.ndim != 2 or points.shape[1] != len(directions) or not np.isfinite(points).all():
        raise ValueError("values must be a finite 2D array with one direction per objective.")
    if set(directions) - {"minimize", "maximize"}:
        raise ValueError("directions must contain only minimize or maximize.")
    factors = np.array([1.0 if direction == "minimize" else -1.0 for direction in directions])
    normalized = points * factors
    return np.array(
        [
            not (np.all(normalized <= point, axis=1) & np.any(normalized < point, axis=1)).any()
            for point in normalized
        ]
    )
