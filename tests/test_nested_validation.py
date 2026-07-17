import unittest

import numpy as np

from adaptive_surrogate import AdaptiveBlackBox


class NestedValidationTest(unittest.TestCase):
    def test_nested_validation_separates_outer_evaluation_from_final_selection(self) -> None:
        x = np.linspace(-1.0, 1.0, 30).reshape(-1, 1)
        engine = AdaptiveBlackBox(epochs=2).fit(
            x,
            x**2,
            validation_folds=2,
            validation_strategy="nested",
        )
        self.assertEqual(engine.validation_strategy, "nested")
        self.assertIn("nrmse", engine.metrics[engine.model_name])
        self.assertIsNotNone(engine.outer_evaluation_metrics)
        self.assertEqual(len(engine.outer_evaluation_metrics["fold_metrics"]), 2)
