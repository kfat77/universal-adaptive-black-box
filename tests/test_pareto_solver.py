import tempfile
import unittest
from pathlib import Path

import numpy as np

from adaptive_surrogate import AdaptiveBlackBox, InverseSolver


class ParetoSolverTest(unittest.TestCase):
    def test_pareto_solver_returns_feasible_nondominated_candidates(self) -> None:
        x = np.linspace(-1.0, 1.0, 50).reshape(-1, 1)
        y = np.column_stack((x[:, 0], x[:, 0] ** 2))
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "model.joblib"
            AdaptiveBlackBox(epochs=3).fit(
                x, y, validation_folds=2, target_names=["benefit", "cost"]
            ).save(artifact)
            front = InverseSolver(str(artifact)).pareto_solve(
                [
                    {"output": "benefit", "direction": "maximize"},
                    {"output": "cost", "direction": "minimize"},
                ],
                [(-1.0, 1.0)],
                n_candidates=40,
                constraints=[lambda value: value[0] >= -0.2],
            )
        self.assertTrue(front)
        self.assertTrue(all(item["non_dominated"] and item["feasible"] for item in front))
