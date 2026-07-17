"""Public-interface tests for bounded inverse solving."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.core_engine import AdaptiveBlackBox
from src.inverse_solver import InverseSolver


class InverseSolverTest(unittest.TestCase):
    def test_returns_distinct_bounded_solutions_with_diagnostics(self) -> None:
        X = np.linspace(-3.0, 3.0, 80).reshape(-1, 1)
        Y = X**2
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "model.joblib"
            AdaptiveBlackBox(epochs=15).fit(X, Y, validation_folds=2).save(artifact)
            results = InverseSolver(str(artifact)).inverse_solve(
                Y_target=np.array([1.0]),
                x_bounds=[(-3.0, 3.0)],
                n_solutions=2,
                min_separation=0.2,
            )

        self.assertEqual(len(results), 2)
        self.assertTrue(all(-3.0 <= result["x"][0] <= 3.0 for result in results))
        self.assertTrue(all("success" in result and "evaluations" in result for result in results))
        self.assertTrue(
            all("optimizer_success" in result and "target_reached" in result for result in results)
        )
        self.assertTrue(
            all("target_error" in result and "weighted_loss" in result for result in results)
        )
        self.assertTrue(
            all(
                "distance_to_training_data" in result and "extrapolation_score" in result
                for result in results
            )
        )
        self.assertTrue(all(result["mse"] < 0.1 for result in results))
        self.assertGreater(abs(results[0]["x"][0] - results[1]["x"][0]), 0.2)

    def test_rejects_non_finite_bounds(self) -> None:
        X = np.linspace(-1.0, 1.0, 20).reshape(-1, 1)
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "model.joblib"
            AdaptiveBlackBox(epochs=2).fit(X, X, validation_folds=2).save(artifact)
            solver = InverseSolver(str(artifact))
            with self.assertRaises(ValueError):
                solver.inverse_solve(np.array([0.0]), [(float("nan"), 1.0)])

    def test_unreachable_target_is_not_marked_successful(self) -> None:
        X = np.linspace(-1.0, 1.0, 30).reshape(-1, 1)
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "model.joblib"
            AdaptiveBlackBox(epochs=2).fit(X, X, validation_folds=2).save(artifact)
            result = InverseSolver(str(artifact)).inverse_solve(
                np.array([3.0]), [(-1.0, 1.0)], target_tolerance=0.01
            )[0]
        self.assertFalse(result["target_reached"])
        self.assertFalse(result["success"])

    def test_constraints_and_fixed_variables_are_reported(self) -> None:
        X = np.linspace(-1.0, 1.0, 40).reshape(-1, 1)
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "model.joblib"
            AdaptiveBlackBox(epochs=5).fit(X, X, validation_folds=2).save(artifact)
            result = InverseSolver(str(artifact)).inverse_solve(
                np.array([0.5]),
                [(-1.0, 1.0)],
                fixed_variables={0: 0.5},
                constraints=[lambda values: values[0] >= 0],
            )[0]
        self.assertAlmostEqual(result["x"][0], 0.5, places=6)
        self.assertEqual(result["constraint_violation"], 0.0)

    def test_affine_constraints_are_enforced_and_reported(self) -> None:
        X = np.linspace(-1.0, 1.0, 40).reshape(-1, 1)
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "model.joblib"
            AdaptiveBlackBox(epochs=3).fit(X, X, validation_folds=2).save(artifact)
            result = InverseSolver(str(artifact)).inverse_solve(
                np.array([0.5]),
                [(-1.0, 1.0)],
                linear_constraints=[{"coefficients": [1.0], "lower": 0.4, "upper": 0.6}],
            )[0]
        self.assertGreaterEqual(result["x"][0], 0.4 - 1e-6)
        self.assertLessEqual(result["x"][0], 0.6 + 1e-6)
        self.assertEqual(result["linear_constraint_violation"], 0.0)

    def test_structured_minimum_and_interval_targets(self) -> None:
        X = np.linspace(-1.0, 1.0, 30).reshape(-1, 1)
        Y = np.column_stack((X[:, 0], X[:, 0] ** 2))
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "model.joblib"
            AdaptiveBlackBox(epochs=3).fit(
                X, Y, validation_folds=2, target_names=["yield", "cost"]
            ).save(artifact)
            result = InverseSolver(str(artifact)).inverse_solve(
                Y_target=np.array([0.0, 0.0]),
                x_bounds=[(-1.0, 1.0)],
                target_spec={"yield": {"min": 0.4}, "cost": {"between": [0.0, 1.0]}},
                target_tolerance=0.02,
            )[0]
        self.assertGreaterEqual(result["predicted_y"][0], 0.38)
        self.assertTrue(result["target_reached"])
