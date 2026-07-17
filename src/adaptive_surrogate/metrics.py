"""Scale-aware regression metrics used for candidate comparison."""

from typing import Sequence

import numpy as np
from sklearn.metrics import r2_score


def validate_output_weights(
    weights: Sequence[float] | np.ndarray | None, output_dim: int
) -> np.ndarray:
    """Return non-negative output weights normalized to sum to one."""
    if weights is None:
        return np.full(output_dim, 1.0 / output_dim)
    array = np.asarray(weights, dtype=float)
    if (
        array.shape != (output_dim,)
        or not np.isfinite(array).all()
        or (array < 0).any()
        or array.sum() <= 0
    ):
        raise ValueError(
            "output_weights must be finite, non-negative, match output dimensions, and sum above zero."
        )
    return array / array.sum()


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_weights: Sequence[float] | np.ndarray | None = None,
    normalization_scales: Sequence[float] | np.ndarray | None = None,
) -> dict[str, float | dict[str, list[float]]]:
    """Compute weighted macro metrics and per-output values in original target units.

    ``normalization_scales`` lets CV folds share a stable nRMSE denominator. If it
    is omitted, the range of ``y_true`` is used for backwards-compatible reporting.
    """
    actual = np.asarray(y_true, dtype=float).reshape(len(y_true), -1)
    predicted = np.asarray(y_pred, dtype=float).reshape(len(y_pred), -1)
    weights = validate_output_weights(output_weights, actual.shape[1])
    mse = np.mean((actual - predicted) ** 2, axis=0)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual - predicted), axis=0)
    if normalization_scales is None:
        scales = np.ptp(actual, axis=0)
    else:
        scales = np.asarray(normalization_scales, dtype=float)
        if (
            scales.shape != (actual.shape[1],)
            or not np.isfinite(scales).all()
            or (scales < 0).any()
        ):
            raise ValueError(
                "normalization_scales must be finite, non-negative, and match outputs."
            )
    scales = np.where(scales > np.finfo(float).eps, scales, 1.0)
    nrmse = rmse / scales
    r2 = np.array(
        [r2_score(actual[:, index], predicted[:, index]) for index in range(actual.shape[1])]
    )
    per_output = {
        "mse": mse.tolist(),
        "rmse": rmse.tolist(),
        "mae": mae.tolist(),
        "r2": r2.tolist(),
        "nrmse": nrmse.tolist(),
    }
    return {
        metric: float(np.dot(values, weights))
        for metric, values in {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "nrmse": nrmse,
        }.items()
    } | {"per_output": per_output}
