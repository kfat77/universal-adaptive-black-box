import unittest

import numpy as np

from adaptive_surrogate import AdaptiveBlackBox


class MonitoringTest(unittest.TestCase):
    def test_ood_assessment_explains_risk_and_distribution_comparison_reports_shift(self) -> None:
        x = np.linspace(-1.0, 1.0, 30).reshape(-1, 1)
        engine = AdaptiveBlackBox(epochs=2).fit(x, x**2, validation_folds=2)
        assessment = engine.assess_distribution([[3.0]])
        self.assertEqual(assessment["risk_level"][0], "high")
        self.assertIn("outside", assessment["explanation"][0])

        report = engine.compare_data_distribution(x, x + 2.0)
        self.assertGreater(report["feature_mean_shift"][0], 1.0)
        self.assertGreater(report["ood_rate"], 0.0)

    def test_engine_active_learning_entrypoint_and_refit(self) -> None:
        x = np.linspace(-1.0, 1.0, 24).reshape(-1, 1)
        engine = AdaptiveBlackBox(epochs=2).fit(x, x, validation_folds=2)
        self.assertEqual(len(engine.recommend_next_experiments([(-1.0, 1.0)], 12, 2)), 2)
        self.assertIs(engine.refit(x, x, validation_folds=2), engine)
