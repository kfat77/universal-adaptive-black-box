"""Tests for cross-validated prediction intervals and OOD assessment."""

import unittest

import numpy as np

from adaptive_surrogate import AdaptiveBlackBox


class UncertaintyAndOODTest(unittest.TestCase):
    def test_prediction_interval_and_distribution_assessment(self) -> None:
        X = np.linspace(-1.0, 1.0, 40).reshape(-1, 1)
        engine = AdaptiveBlackBox(epochs=5).fit(X, X**2, validation_folds=2)
        prediction, lower, upper = engine.predict_interval([[0.2]], confidence=0.9)
        assessment = engine.assess_distribution([[0.2], [10.0]])
        self.assertTrue(np.all(lower <= prediction))
        self.assertTrue(np.all(prediction <= upper))
        self.assertTrue(assessment["in_distribution"][0])
        self.assertFalse(assessment["in_distribution"][1])
