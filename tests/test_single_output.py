"""Regression test for random-forest prediction with one output column."""

import unittest

import numpy as np

from src.core_engine import AdaptiveBlackBox


class SingleOutputRegressionTest(unittest.TestCase):
    def test_fit_and_predict_keep_single_output_two_dimensional(self) -> None:
        X = np.linspace(-1.0, 1.0, 24).reshape(-1, 1)
        Y = (2.0 * X + 0.1).reshape(-1, 1)
        model = AdaptiveBlackBox(epochs=1).fit(X, Y)
        prediction = model.predict(np.array([[0.25]]))
        self.assertEqual(prediction.shape, (1, 1))


if __name__ == "__main__":
    unittest.main()
