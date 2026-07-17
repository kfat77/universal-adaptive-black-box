"""Tests for scale-aware regression evaluation."""

import unittest

import numpy as np

from adaptive_surrogate import AdaptiveBlackBox
from src.metrics import compute_regression_metrics, validate_output_weights


class RegressionMetricsTest(unittest.TestCase):
    def test_normalized_metrics_do_not_favor_large_scale_outputs(self) -> None:
        y_true = np.array([[0.0, 0.0], [1.0, 1000.0], [2.0, 2000.0]])
        y_pred = np.array([[0.0, 100.0], [1.0, 1100.0], [2.0, 2100.0]])
        metrics = compute_regression_metrics(y_true, y_pred)
        self.assertAlmostEqual(metrics["nrmse"], 0.025, places=6)
        self.assertEqual(len(metrics["per_output"]["nrmse"]), 2)

    def test_output_weights_are_normalized_and_validated(self) -> None:
        np.testing.assert_allclose(validate_output_weights([2.0, 1.0], 2), [2 / 3, 1 / 3])
        with self.assertRaises(ValueError):
            validate_output_weights([1.0], 2)
        with self.assertRaises(ValueError):
            validate_output_weights([0.0, 0.0], 2)

    def test_named_output_weights_follow_target_names(self) -> None:
        x = np.linspace(-1.0, 1.0, 20).reshape(-1, 1)
        y = np.column_stack((x[:, 0], x[:, 0] ** 2))
        engine = AdaptiveBlackBox(epochs=2).fit(
            x,
            y,
            validation_folds=2,
            target_names=["yield", "cost"],
            output_weights={"cost": 1.0, "yield": 3.0},
        )
        np.testing.assert_allclose(engine.output_weights, [0.75, 0.25])
