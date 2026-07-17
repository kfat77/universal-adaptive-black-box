"""Inspect residual intervals, OOD flags, and experimental recommendations."""

import numpy as np

from adaptive_surrogate import AdaptiveBlackBox, recommend_next_experiments


def main() -> None:
    x_train = np.linspace(-2.0, 2.0, 100).reshape(-1, 1)
    y_train = np.cos(2.0 * x_train)
    engine = AdaptiveBlackBox(epochs=120).fit(x_train, y_train)

    prediction, lower, upper = engine.predict_interval([[1.0]], confidence=0.90)
    assessment = engine.assess_distribution([[3.5]])
    recommendations = recommend_next_experiments(
        engine,
        x_bounds=[(-3.0, 3.0)],
        n_candidates=128,
        n_recommendations=3,
        strategy="uncertainty_diversity",
    )

    print("90% residual interval:", lower.ravel()[0], prediction.ravel()[0], upper.ravel()[0])
    print("x=3.5 is in distribution:", bool(assessment["in_distribution"][0]))
    print("Recommended next inputs:", [candidate["x"][0] for candidate in recommendations])


if __name__ == "__main__":
    main()
