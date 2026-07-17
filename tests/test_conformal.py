import unittest

import numpy as np

from adaptive_surrogate import AdaptiveBlackBox


class ConformalTest(unittest.TestCase):
    def test_split_conformal_uses_a_disjoint_calibration_set(self) -> None:
        x = np.linspace(-2.0, 2.0, 40).reshape(-1, 1)
        engine = AdaptiveBlackBox(epochs=2).fit(
            x,
            x**2,
            validation_folds=2,
            uncertainty_method="split_conformal",
            calibration_fraction=0.25,
        )
        prediction, lower, upper = engine.predict_interval([[0.2]], confidence=0.9)
        self.assertEqual(engine.uncertainty_method, "split_conformal")
        self.assertEqual(engine.calibration_samples, 10)
        self.assertEqual(engine.training_samples, 30)
        self.assertEqual(prediction.shape, lower.shape)
        self.assertEqual(prediction.shape, upper.shape)

    def test_split_conformal_rejects_too_few_observations(self) -> None:
        x = np.arange(8.0).reshape(-1, 1)
        with self.assertRaisesRegex(ValueError, "at least 10"):
            AdaptiveBlackBox(epochs=1).fit(x, x, uncertainty_method="split_conformal")

    def test_split_conformal_uses_finite_sample_order_statistic(self) -> None:
        x = np.linspace(-1.0, 1.0, 20).reshape(-1, 1)
        engine = AdaptiveBlackBox(epochs=1).fit(x, x, validation_folds=2)
        engine.uncertainty_method = "split_conformal"
        engine.calibration_residuals = np.array([[1.0], [2.0], [3.0]])
        prediction, _, upper = engine.predict_interval([[0.0]], confidence=0.75)
        np.testing.assert_allclose(upper - prediction, [[3.0]])
