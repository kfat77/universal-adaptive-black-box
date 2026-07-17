"""Regression test for random-forest prediction with one output column."""

import unittest

import numpy as np
import pandas as pd

from src.core_engine import AdaptiveBlackBox


class SingleOutputRegressionTest(unittest.TestCase):
    def test_fit_and_predict_keep_single_output_two_dimensional(self) -> None:
        X = np.linspace(-1.0, 1.0, 24).reshape(-1, 1)
        Y = (2.0 * X + 0.1).reshape(-1, 1)
        model = AdaptiveBlackBox(epochs=1).fit(X, Y)
        prediction = model.predict(np.array([[0.25]]))
        self.assertEqual(prediction.shape, (1, 1))

    def test_training_reports_cross_validation_summary(self) -> None:
        X = np.linspace(-1.0, 1.0, 30).reshape(-1, 1)
        Y = (X**2).reshape(-1, 1)
        model = AdaptiveBlackBox(epochs=2).fit(X, Y, validation_folds=3)
        self.assertEqual(
            set(model.metrics),
            {
                "dummy",
                "linear_regression",
                "ridge",
                "mlp",
                "random_forest",
                "extra_trees",
                "hist_gradient_boosting",
            },
        )
        self.assertEqual(model.training_samples, len(X))
        self.assertIn("mse_std", model.metrics[model.model_name])
        self.assertIn("r2_std", model.metrics[model.model_name])

    def test_multi_output_training_and_prediction(self) -> None:
        X = np.linspace(-1.0, 1.0, 30).reshape(-1, 1)
        Y = np.column_stack((X.ravel() ** 2, 2.0 * X.ravel() + 1.0))
        model = AdaptiveBlackBox(epochs=2).fit(X, Y, validation_folds=3)
        self.assertEqual(model.predict(np.array([[0.25]])).shape, (1, 2))

    def test_mlp_configuration_supports_minibatches_and_early_stopping(self) -> None:
        X = np.linspace(-1.0, 1.0, 40).reshape(-1, 1)
        model = AdaptiveBlackBox(
            mlp_config={"hidden_layers": (16,), "batch_size": 8, "max_epochs": 5, "patience": 2}
        )
        model.fit(X, X, validation_folds=2)
        self.assertEqual(model.mlp_config["batch_size"], 8)

    def test_mlp_scheduler_configuration_is_supported(self) -> None:
        X = np.linspace(-1.0, 1.0, 20).reshape(-1, 1)
        model = AdaptiveBlackBox(
            mlp_config={"max_epochs": 3, "patience": 1, "scheduler_patience": 1}
        ).fit(X, X, validation_folds=2)
        self.assertEqual(model.mlp_config["scheduler_patience"], 1)

    def test_dataframe_prediction_reorders_known_columns_and_rejects_schema_drift(self) -> None:
        X = pd.DataFrame({"temperature": np.linspace(-1.0, 1.0, 24), "pressure": 2.0})
        Y = pd.DataFrame({"yield": 2.0 * X["temperature"] + X["pressure"]})
        model = AdaptiveBlackBox(epochs=2).fit(X, Y, validation_folds=2)
        ordered = model.predict(pd.DataFrame({"temperature": [0.2], "pressure": [2.0]}))
        reversed_columns = model.predict(pd.DataFrame({"pressure": [2.0], "temperature": [0.2]}))
        np.testing.assert_allclose(ordered, reversed_columns)
        with self.assertRaisesRegex(ValueError, "missing"):
            model.predict(pd.DataFrame({"temperature": [0.2]}))


if __name__ == "__main__":
    unittest.main()
