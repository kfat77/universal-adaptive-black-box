"""Tests for scale-aware regression evaluation."""

import unittest

import numpy as np

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
