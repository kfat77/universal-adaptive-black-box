"""Lightweight model-agnostic explanation utilities for trained surrogates."""

import numpy as np

from .core_engine import AdaptiveBlackBox


def permutation_importance(
    engine: AdaptiveBlackBox,
    X: np.ndarray,
    Y: np.ndarray,
    n_repeats: int = 5,
    random_state: int = 42,
) -> list[dict[str, float | str]]:
    """Estimate feature importance from prediction-error increase after permutation.

    Scores describe predictive association in the supplied data, never causal effect.
    """
    features = engine._as_2d(X, "X")
    targets = engine._as_2d(Y, "Y")
    if features.shape[0] != targets.shape[0] or features.shape[1] != engine.input_dim:
        raise ValueError("X and Y must have matching rows and the trained input dimension.")
    if n_repeats < 1:
        raise ValueError("n_repeats must be positive.")
    baseline = float(np.mean((engine.predict(features) - targets) ** 2))
    rng = np.random.default_rng(random_state)
    names = engine.feature_names or tuple(f"feature_{index}" for index in range(features.shape[1]))
    results: list[dict[str, float | str]] = []
    for index, name in enumerate(names):
        increases = []
        for _ in range(n_repeats):
            permuted = features.copy()
            permuted[:, index] = rng.permutation(permuted[:, index])
            increases.append(float(np.mean((engine.predict(permuted) - targets) ** 2) - baseline))
        results.append(
            {
                "feature": name,
                "importance_mean": float(np.mean(increases)),
                "importance_std": float(np.std(increases)),
            }
        )
    return sorted(results, key=lambda result: float(result["importance_mean"]), reverse=True)


def local_sensitivity(
    engine: AdaptiveBlackBox, X: np.ndarray, relative_step: float = 1e-4
) -> np.ndarray:
    """Return finite-difference output sensitivity for each sample and feature."""
    values = engine._as_2d(X, "X")
    if values.shape[1] != engine.input_dim or relative_step <= 0:
        raise ValueError("X must match the trained dimension and relative_step must be positive.")
    if engine.output_dim is None or engine.input_dim is None:
        raise RuntimeError("Train or load the engine before computing sensitivity.")
    output_dim, input_dim = engine.output_dim, engine.input_dim
    gradients = np.empty((len(values), output_dim, input_dim))
    for feature in range(input_dim):
        step = relative_step * np.maximum(np.abs(values[:, feature]), 1.0)
        plus, minus = values.copy(), values.copy()
        plus[:, feature] += step
        minus[:, feature] -= step
        gradients[:, :, feature] = (engine.predict(plus) - engine.predict(minus)) / (
            2.0 * step[:, None]
        )
    return gradients
