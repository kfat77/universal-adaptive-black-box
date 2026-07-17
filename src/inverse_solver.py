"""Numerical inverse solving for forward models that do not have analytic inverses."""

from typing import Any

import numpy as np
from scipy.optimize import differential_evolution, minimize

from src.core_engine import AdaptiveBlackBox


class InverseSolver:
    def __init__(self, artifact_path: str):
        self.model = AdaptiveBlackBox.load(artifact_path)

    def inverse_solve(self, Y_target: np.ndarray, x_bounds: list[tuple[float, float]],
                      n_solutions: int = 1, min_separation: float = 1e-6,
                      max_attempts: int | None = None) -> list[dict[str, Any]]:
        """Find inputs whose forward prediction is close to ``Y_target``.

        A black-box need not be one-to-one, so this searches instead of algebraically
        inverting it. Differential evolution first explores the bounded global input
        space. L-BFGS-B then locally refines each search result by minimizing squared
        output error. Multiple restarts can expose alternative valid inputs.
        ``min_separation`` is Euclidean distance in the original input units.
        """
        target = np.asarray(Y_target, dtype=float).reshape(1, -1)
        if not np.isfinite(target).all() or target.shape[1] != self.model.output_dim:
            raise ValueError(f"Y_target must contain {self.model.output_dim} values.")
        try:
            bounds = [(float(low), float(high)) for low, high in x_bounds]
        except (TypeError, ValueError):
            raise ValueError("x_bounds must contain numeric (low, high) pairs.") from None
        if (len(bounds) != self.model.input_dim
                or any(not np.isfinite(low) or not np.isfinite(high) or low >= high for low, high in bounds)):
            raise ValueError("x_bounds must provide (low, high) with low < high for every input.")
        if n_solutions < 1 or min_separation < 0:
            raise ValueError("n_solutions must be positive and min_separation cannot be negative.")

        def objective(x: np.ndarray) -> float:
            residual = self.model.predict(x.reshape(1, -1)) - target
            return float(np.mean(residual ** 2))

        answers: list[dict[str, Any]] = []
        attempts = max_attempts if max_attempts is not None else n_solutions * 5
        if attempts < n_solutions:
            raise ValueError("max_attempts must be at least n_solutions.")
        for attempt in range(attempts):
            global_result = differential_evolution(
                objective, bounds, seed=self.model.random_state + attempt, polish=False
            )
            local_result = minimize(objective, global_result.x, bounds=bounds, method="L-BFGS-B")
            x_solution = local_result.x
            if any(np.linalg.norm(x_solution - answer["x"]) < min_separation for answer in answers):
                continue
            answers.append({
                "x": x_solution,
                "predicted_y": self.model.predict(x_solution[None, :])[0],
                "mse": objective(x_solution),
                "success": bool(global_result.success and local_result.success),
                "evaluations": int(global_result.nfev + local_result.nfev),
                "attempt": attempt + 1,
                "message": str(local_result.message),
            })
            if len(answers) == n_solutions:
                break
        return sorted(answers, key=lambda answer: float(answer["mse"]))
