"""Search a bounded input domain for values that meet an output target."""

from pathlib import Path

import numpy as np

from adaptive_surrogate import AdaptiveBlackBox, InverseSolver


def main() -> None:
    x_train = np.linspace(-3.0, 3.0, 160).reshape(-1, 1)
    y_train = np.sin(x_train)
    engine = AdaptiveBlackBox(epochs=120).fit(x_train, y_train)

    artifact = Path("artifacts/constrained_inverse_example.joblib")
    artifact.parent.mkdir(exist_ok=True)
    engine.save(artifact)
    solver = InverseSolver(str(artifact))

    # Callable constraints return True/False or a non-negative feasible margin.
    solutions = solver.inverse_solve(
        Y_target=[0.5],
        x_bounds=[(-3.0, 3.0)],
        target_tolerance=0.02,
        constraints=[lambda x: x[0] >= 0.0],
        reference_x=[1.0],
        distance_penalty=0.05,
        n_solutions=2,
    )
    for solution in solutions:
        print(
            "x=",
            solution["x"],
            "predicted_y=",
            solution["predicted_y"],
            "success=",
            solution["success"],
            "in_distribution=",
            solution["in_distribution"],
        )


if __name__ == "__main__":
    main()
