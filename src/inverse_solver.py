"""Numerical inverse solving for forward models that do not have analytic inverses."""

import numpy as np
from scipy.optimize import differential_evolution, minimize

from src.core_engine import AdaptiveBlackBox


class InverseSolver:
    def __init__(self, artifact_path: str):
        self.model = AdaptiveBlackBox.load(artifact_path)

    def inverse_solve(self, Y_target: np.ndarray, x_bounds: list[tuple[float, float]],
                      n_solutions: int = 1) -> list[dict[str, np.ndarray | float]]:
        """Find inputs whose forward prediction is close to ``Y_target``.

        A black-box need not be one-to-one, so this searches instead of algebraically
        inverting it. Differential evolution first explores the bounded global input
        space. L-BFGS-B then locally refines each search result by minimizing squared
        output error. Multiple restarts can expose alternative valid inputs.
        """
        target = np.asarray(Y_target, dtype=float).reshape(1, -1)
        if target.shape[1] != self.model.output_dim:
            raise ValueError(f"Y_target must contain {self.model.output_dim} values.")
        if len(x_bounds) != self.model.input_dim or any(low >= high for low, high in x_bounds):
            raise ValueError("x_bounds must provide (low, high) with low < high for every input.")

        def objective(x: np.ndarray) -> float:
            residual = self.model.predict(x.reshape(1, -1)) - target
            return float(np.mean(residual ** 2))

        answers = []
        for seed in range(n_solutions):
            global_result = differential_evolution(objective, x_bounds, seed=seed, polish=False)
            local_result = minimize(objective, global_result.x, bounds=x_bounds, method="L-BFGS-B")
            x_solution = local_result.x
            answers.append({"x": x_solution, "predicted_y": self.model.predict(x_solution[None, :])[0],
                            "mse": objective(x_solution)})
        return sorted(answers, key=lambda answer: float(answer["mse"]))
