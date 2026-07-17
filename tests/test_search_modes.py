import unittest
from unittest.mock import patch

import numpy as np
from sklearn.linear_model import Ridge

from adaptive_surrogate import AdaptiveBlackBox


class SearchModeTest(unittest.TestCase):
    def test_balanced_search_records_reproducible_inner_search_details(self) -> None:
        x = np.linspace(-1.0, 1.0, 24).reshape(-1, 1)
        engine = AdaptiveBlackBox(epochs=2).fit(x, x**2, validation_folds=2, search_mode="balanced")
        self.assertEqual(engine.search_mode, "balanced")
        self.assertIn("ridge", engine.search_details)
        self.assertEqual(engine.search_details["ridge"]["budget"], 3)

    def test_unknown_search_mode_is_rejected(self) -> None:
        x = np.linspace(-1.0, 1.0, 12).reshape(-1, 1)
        with self.assertRaisesRegex(ValueError, "search_mode"):
            AdaptiveBlackBox(epochs=1).fit(x, x, search_mode="slow")

    def test_multi_output_gradient_boosting_search_uses_wrapped_parameters(self) -> None:
        x = np.linspace(-1.0, 1.0, 12).reshape(-1, 1)
        y = np.column_stack((x[:, 0], x[:, 0] ** 2))
        engine = AdaptiveBlackBox(epochs=1)
        engine.output_dim = 2
        engine.search_details = {}
        model = engine._fit_candidate(
            "hist_gradient_boosting",
            x,
            y,
            seed=7,
            search_mode="balanced",
            selection_metric="mae",
            output_weights=np.array([0.75, 0.25]),
            normalization_scales=np.array([2.0, 1.0]),
        )
        self.assertIn("hist_gradient_boosting", engine.search_details)
        self.assertEqual(engine.search_details["hist_gradient_boosting"]["selection_metric"], "mae")
        np.testing.assert_allclose(
            engine.search_details["hist_gradient_boosting"]["output_weights"], [0.75, 0.25]
        )
        self.assertTrue(
            all(
                key.startswith("estimator__")
                for key in engine.search_details["hist_gradient_boosting"]["best_params"]
            )
        )
        self.assertEqual(model.predict(x).shape, y.shape)

    def test_search_scorer_restores_original_target_units(self) -> None:
        x = np.arange(12.0).reshape(-1, 1)
        y_scaled = np.column_stack((x[:, 0], x[:, 0]))
        captured: dict[str, object] = {}

        class CapturingSearch:
            def __init__(self, estimator, parameter_space, **kwargs) -> None:
                captured["scorer"] = kwargs["scoring"]
                self.estimator = estimator
                self.best_params_ = {"alpha": 1.0}
                self.best_score_ = 0.0

            def fit(self, values, targets, **kwargs):
                self.best_estimator_ = self.estimator.fit(values, targets)
                return self

        engine = AdaptiveBlackBox(epochs=1)
        engine.output_dim = 2
        engine.search_details = {}
        with patch("adaptive_surrogate.core_engine.RandomizedSearchCV", CapturingSearch):
            engine._fit_candidate(
                "ridge",
                x,
                y_scaled,
                seed=7,
                search_mode="balanced",
                selection_metric="mae",
                output_weights=np.array([0.25, 0.75]),
                normalization_scales=np.array([2.0, 20.0]),
                target_mean=np.array([10.0, 100.0]),
                target_scale=np.array([2.0, 20.0]),
            )
        scorer = captured["scorer"]
        model = Ridge().fit(x[:8], y_scaled[:8])
        actual = y_scaled[8:]
        expected = -float(np.average([2.0, 20.0], weights=[0.25, 0.75]))
        with patch.object(model, "predict", return_value=actual + 1.0):
            self.assertAlmostEqual(scorer(model, x[8:], actual), expected)
        self.assertEqual(engine.search_details["ridge"]["scoring_target_units"], "original")
