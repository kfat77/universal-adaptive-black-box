"""Tests for experiment recommendation and Pareto filtering."""

import unittest

import numpy as np

from adaptive_surrogate import AdaptiveBlackBox, non_dominated_mask, recommend_next_experiments


class ActiveLearningAndParetoTest(unittest.TestCase):
    def test_recommendations_stay_within_bounds(self) -> None:
        X = np.linspace(-1.0, 1.0, 30).reshape(-1, 1)
        engine = AdaptiveBlackBox(epochs=3).fit(X, X**2, validation_folds=2)
        recommendations = recommend_next_experiments(engine, [(-2.0, 2.0)], 20, 3)
        self.assertEqual(len(recommendations), 3)
        self.assertTrue(all(-2.0 <= item["x"][0] <= 2.0 for item in recommendations))

    def test_pareto_filter_respects_objective_directions(self) -> None:
        values = np.array([[1.0, 5.0], [2.0, 3.0], [3.0, 2.0], [4.0, 6.0]])
        mask = non_dominated_mask(values, ["minimize", "maximize"])
        np.testing.assert_array_equal(mask, [True, False, False, True])

    def test_helpers_reject_non_finite_values(self) -> None:
        X = np.linspace(-1.0, 1.0, 20).reshape(-1, 1)
        engine = AdaptiveBlackBox(epochs=2).fit(X, X, validation_folds=2)
        with self.assertRaises(ValueError):
            recommend_next_experiments(engine, [(float("nan"), 1.0)])
        with self.assertRaises(ValueError):
            non_dominated_mask([[0.0, float("inf")]], ["minimize", "maximize"])
