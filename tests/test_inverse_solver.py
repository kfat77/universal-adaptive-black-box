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
                Y_target=np.array([1.0]), x_bounds=[(-3.0, 3.0)], n_solutions=2,
                min_separation=0.2,
            )

        self.assertEqual(len(results), 2)
        self.assertTrue(all(-3.0 <= result["x"][0] <= 3.0 for result in results))
        self.assertTrue(all("success" in result and "evaluations" in result for result in results))
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
