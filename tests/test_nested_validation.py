import unittest
from unittest.mock import patch

import numpy as np

from adaptive_surrogate import AdaptiveBlackBox, core_engine


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

    def test_nested_group_validation_preserves_groups_inside_inner_selection(self) -> None:
        x = np.linspace(-1.0, 1.0, 24).reshape(-1, 1)
        groups = np.repeat(np.arange(6), 4)
        strategies: list[str] = []
        original_build_splits = core_engine.build_splits

        def record_strategy(*args, **kwargs):
            strategies.append(args[0])
            return original_build_splits(*args, **kwargs)

        with patch("adaptive_surrogate.core_engine.build_splits", side_effect=record_strategy):
            engine = AdaptiveBlackBox(epochs=1).fit(
                x, x**2, validation_folds=2, validation_strategy="nested", groups=groups
            )
        self.assertEqual(engine.validation_strategy, "nested")
        self.assertEqual(set(strategies), {"group_kfold"})
