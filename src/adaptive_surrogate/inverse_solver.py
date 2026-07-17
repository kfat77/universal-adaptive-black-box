"""Numerical inverse solving for forward models that do not have analytic inverses."""

from typing import Any, Callable

import numpy as np
from scipy.optimize import differential_evolution, minimize

from .core_engine import AdaptiveBlackBox


class InverseSolver:
    def __init__(self, artifact_path: str):
        self.model = AdaptiveBlackBox.load(artifact_path)

    def inverse_solve(
        self,
        Y_target: np.ndarray,
        x_bounds: list[tuple[float, float]],
        n_solutions: int = 1,
        min_separation: float = 1e-6,
        max_attempts: int | None = None,
        target_tolerance: float | np.ndarray = 1e-4,
        distance_metric: str = "normalized_euclidean",
        input_weights: np.ndarray | None = None,
        constraints: list[Callable[[np.ndarray], bool | float]] | None = None,
        fixed_variables: dict[int, float] | None = None,
        reference_x: np.ndarray | None = None,
        distance_penalty: float = 0.0,
        ood_penalty: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Find inputs whose forward prediction is close to ``Y_target``.

        A black-box need not be one-to-one, so this searches instead of algebraically
        inverting it. Differential evolution first explores the bounded global input
        space. L-BFGS-B then locally refines each search result by minimizing squared
        output error. Multiple restarts can expose alternative valid inputs.
        ``min_separation`` defaults to Euclidean distance after every input is
        normalized by its supplied bounds. ``target_tolerance`` defines whether
        each output is close enough to the requested target.
        """
        target = np.asarray(Y_target, dtype=float).reshape(1, -1)
        if not np.isfinite(target).all() or target.shape[1] != self.model.output_dim:
            raise ValueError(f"Y_target must contain {self.model.output_dim} values.")
        try:
            bounds = [(float(low), float(high)) for low, high in x_bounds]
        except (TypeError, ValueError):
            raise ValueError("x_bounds must contain numeric (low, high) pairs.") from None
        if len(bounds) != self.model.input_dim or any(
            not np.isfinite(low) or not np.isfinite(high) or low >= high for low, high in bounds
        ):
            raise ValueError("x_bounds must provide (low, high) with low < high for every input.")
        if n_solutions < 1 or min_separation < 0:
            raise ValueError("n_solutions must be positive and min_separation cannot be negative.")
        tolerances = np.asarray(target_tolerance, dtype=float)
        if tolerances.ndim == 0:
            tolerances = np.full(self.model.output_dim, float(tolerances))
        if (
            tolerances.shape != (self.model.output_dim,)
            or not np.isfinite(tolerances).all()
            or (tolerances < 0).any()
        ):
            raise ValueError(
                "target_tolerance must be a finite non-negative scalar or one value per output."
            )
        if distance_metric not in {
            "normalized_euclidean",
            "original_euclidean",
            "weighted_euclidean",
        }:
            raise ValueError(
                "distance_metric must be normalized_euclidean, original_euclidean, or weighted_euclidean."
            )
        weights = (
            np.ones(self.model.input_dim)
            if input_weights is None
            else np.asarray(input_weights, dtype=float)
        )
        if (
            weights.shape != (self.model.input_dim,)
            or not np.isfinite(weights).all()
            or (weights < 0).any()
            or weights.sum() <= 0
        ):
            raise ValueError(
                "input_weights must be finite, non-negative, match input dimensions, and sum above zero."
            )
        widths = np.array([bound[1] - bound[0] for bound in bounds])
        fixed = fixed_variables or {}
        if any(
            index < 0 or index >= self.model.input_dim or not np.isfinite(value)
            for index, value in fixed.items()
        ):
            raise ValueError("fixed_variables must map valid input indices to finite values.")
        if any(not bounds[index][0] <= value <= bounds[index][1] for index, value in fixed.items()):
            raise ValueError("fixed variable values must lie within x_bounds.")
        if distance_penalty < 0 or ood_penalty < 0:
            raise ValueError("distance_penalty and ood_penalty must be non-negative.")
        reference = None if reference_x is None else np.asarray(reference_x, dtype=float)
        if reference is not None and (
            reference.shape != (self.model.input_dim,) or not np.isfinite(reference).all()
        ):
            raise ValueError("reference_x must contain one finite value per input.")
        effective_bounds: list[tuple[float, float]] = []
        for index, bound in enumerate(bounds):
            if index in fixed:
                value = fixed[index]
                effective_bounds.append((value - 1e-12, value + 1e-12))
            else:
                effective_bounds.append(bound)

        def apply_fixed(x: np.ndarray) -> np.ndarray:
            candidate = np.asarray(x, dtype=float).copy()
            for index, value in fixed.items():
                candidate[index] = value
            return candidate

        def constraint_violation(x: np.ndarray) -> float:
            violation = 0.0
            for constraint in constraints or []:
                result = constraint(x)
                if isinstance(result, (bool, np.bool_)):
                    violation += 0.0 if result else 1.0
                else:
                    value = float(result)
                    if not np.isfinite(value):
                        return float("inf")
                    violation += max(0.0, -value)
            return violation

        def objective(x: np.ndarray) -> float:
            candidate = apply_fixed(x)
            residual = self.model.predict(candidate.reshape(1, -1)) - target
            target_loss = float(
                np.average(residual.reshape(-1) ** 2, weights=self.model.output_weights)
            )
            violation = constraint_violation(candidate)
            reference_loss = (
                0.0 if reference is None else solution_distance(candidate, reference) ** 2
            )
            ood_loss = float(
                self.model.assess_distribution(candidate[None, :])["extrapolation_score"][0]
            )
            return (
                target_loss
                + 1e6 * violation
                + distance_penalty * reference_loss
                + ood_penalty * ood_loss
            )

        def solution_distance(first: np.ndarray, second: np.ndarray) -> float:
            difference = first - second
            if distance_metric != "original_euclidean":
                difference = difference / widths
            if distance_metric == "weighted_euclidean":
                return float(np.sqrt(np.average(difference**2, weights=weights)))
            return float(np.linalg.norm(difference))

        answers: list[dict[str, Any]] = []
        attempts = max_attempts if max_attempts is not None else n_solutions * 5
        if attempts < n_solutions:
            raise ValueError("max_attempts must be at least n_solutions.")
        for attempt in range(attempts):
            global_result = differential_evolution(
                objective, effective_bounds, seed=self.model.random_state + attempt, polish=False
            )
            local_result = minimize(
                objective, global_result.x, bounds=effective_bounds, method="L-BFGS-B"
            )
            x_solution = apply_fixed(local_result.x)
            if any(
                solution_distance(x_solution, answer["x"]) < min_separation for answer in answers
            ):
                continue
            prediction = self.model.predict(x_solution[None, :])[0]
            assessment = self.model.assess_distribution(x_solution[None, :])
            target_error = np.abs(prediction - target[0])
            target_loss = float(
                np.average((prediction - target[0]) ** 2, weights=self.model.output_weights)
            )
            violation = constraint_violation(x_solution)
            reference_loss = (
                0.0 if reference is None else solution_distance(x_solution, reference) ** 2
            )
            ood_loss = float(assessment["extrapolation_score"][0])
            weighted_loss = (
                target_loss
                + 1e6 * violation
                + distance_penalty * reference_loss
                + ood_penalty * ood_loss
            )
            optimizer_success = bool(global_result.success and local_result.success)
            target_reached = bool(np.all(target_error <= tolerances))
            answers.append(
                {
                    "x": x_solution,
                    "predicted_y": prediction,
                    "target_error": target_error,
                    "mse": float(np.mean((prediction - target[0]) ** 2)),
                    "weighted_loss": weighted_loss,
                    "constraint_violation": violation,
                    "reference_distance": reference_loss**0.5,
                    "ood_penalty_component": ood_penalty * ood_loss,
                    "optimizer_success": optimizer_success,
                    "target_reached": target_reached,
                    "feasible": bool(violation == 0.0),
                    "in_distribution": bool(assessment["in_distribution"][0]),
                    "success": bool(
                        optimizer_success
                        and target_reached
                        and violation == 0.0
                        and assessment["in_distribution"][0]
                    ),
                    "distance_to_training_data": float(assessment["nearest_training_distance"][0]),
                    "extrapolation_score": float(assessment["extrapolation_score"][0]),
                    "evaluations": int(global_result.nfev + local_result.nfev),
                    "attempt": attempt + 1,
                    "message": str(local_result.message),
                }
            )
            if len(answers) == n_solutions:
                break
        return sorted(answers, key=lambda answer: float(answer["weighted_loss"]))
