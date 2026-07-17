import unittest

import numpy as np

from adaptive_surrogate import AdaptiveBlackBox, local_sensitivity, permutation_importance


class ExplainabilityTest(unittest.TestCase):
    def test_permutation_importance_and_local_sensitivity(self) -> None:
        rng = np.random.default_rng(3)
        x = rng.normal(size=(40, 2))
        y = 3.0 * x[:, [0]] + 0.1 * x[:, [1]]
        engine = AdaptiveBlackBox(epochs=3).fit(
            x, y, validation_folds=2, feature_names=["signal", "noise"]
        )
        importance = permutation_importance(engine, x, y, n_repeats=2)
        sensitivity = local_sensitivity(engine, x[:2])
        self.assertEqual(importance[0]["feature"], "signal")
        self.assertEqual(sensitivity.shape, (2, 1, 2))
