"""Lightweight experiment recommendation from uncertainty and diversity scores."""

import numpy as np

from .core_engine import AdaptiveBlackBox


def recommend_next_experiments(
    engine: AdaptiveBlackBox,
    x_bounds: list[tuple[float, float]],
    n_candidates: int = 256,
    n_recommendations: int = 5,
    strategy: str = "uncertainty_diversity",
    random_state: int = 42,
) -> list[dict[str, np.ndarray | float | str]]:
    """Recommend bounded candidate inputs; this never executes physical experiments."""
    if engine.input_dim is None:
        raise RuntimeError("Train or load the engine before requesting experiments.")
    if strategy not in {"uncertainty", "diversity", "uncertainty_diversity"}:
        raise ValueError("strategy must be uncertainty, diversity, or uncertainty_diversity.")
    bounds = np.asarray(x_bounds, dtype=float)
    if (
        bounds.shape != (engine.input_dim, 2)
        or not np.isfinite(bounds).all()
        or np.any(bounds[:, 0] >= bounds[:, 1])
    ):
        raise ValueError("x_bounds must contain valid lower and upper bounds for every input.")
    if not 1 <= n_recommendations <= n_candidates:
        raise ValueError("n_recommendations must be between 1 and n_candidates.")
    rng = np.random.default_rng(random_state)
    candidates = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_candidates, engine.input_dim))
    prediction, lower, upper = engine.predict_interval(candidates)
    assessment = engine.assess_distribution(candidates)
    diversity = assessment["nearest_training_distance"]
    # Residual intervals are globally calibrated and can have equal width. Scale
    # them by novelty to form a varying uncertainty proxy without claiming a new
    # conformal coverage guarantee.
    interval_width = np.mean(upper - lower, axis=1)
    uncertainty = interval_width * (1.0 + assessment["extrapolation_score"])
    uncertainty_score = uncertainty / max(float(uncertainty.max()), np.finfo(float).eps)
    diversity_score = diversity / max(float(diversity.max()), np.finfo(float).eps)
    score = uncertainty_score if strategy == "uncertainty" else diversity_score
    if strategy == "uncertainty_diversity":
        score = (uncertainty_score + diversity_score) / 2
    selected = np.argsort(score)[-n_recommendations:][::-1]
    return [
        {
            "x": candidates[index],
            "predicted_y": prediction[index],
            "lower": lower[index],
            "upper": upper[index],
            "uncertainty_score": float(uncertainty_score[index]),
            "diversity_score": float(diversity_score[index]),
            "ood_score": float(assessment["extrapolation_score"][index]),
            "rationale": f"{strategy}; uncertainty is interval-width times training-domain novelty",
        }
        for index in selected
    ]
