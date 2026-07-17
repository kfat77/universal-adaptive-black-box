"""End-to-end example: fit noisy sine data, predict it, then solve backwards."""

from pathlib import Path

import numpy as np

from adaptive_surrogate import AdaptiveBlackBox, ForwardSolver, InverseSolver


def main() -> None:
    rng = np.random.default_rng(7)
    X = rng.uniform(-6.0, 6.0, size=(600, 1))
    Y = np.sin(X) + 0.06 * rng.normal(size=(600, 1))
    artifact = Path("artifacts/adaptive_black_box.joblib")

    engine = AdaptiveBlackBox(epochs=500).fit(X, Y)
    engine.save(artifact)
    print("Selected model:", engine.model_name)
    print("Validation nRMSE:", engine.metrics[engine.model_name]["nrmse"])

    forward = ForwardSolver(str(artifact))
    print("Forward prediction for x=1.0:", forward.predict(np.array([[1.0]])))

    inverse = InverseSolver(str(artifact))
    solutions = inverse.inverse_solve(
        Y_target=np.array([0.5]), x_bounds=[(-6.0, 6.0)], n_solutions=2
    )
    for index, solution in enumerate(solutions, start=1):
        print(
            f"Inverse solution {index}: x={solution['x']}, "
            f"predicted_y={solution['predicted_y']}, mse={solution['mse']:.6g}, "
            f"success={solution['success']}"
        )


if __name__ == "__main__":
    main()
