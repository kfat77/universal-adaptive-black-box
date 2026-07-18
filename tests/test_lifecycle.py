"""Public-interface tests for evaluation, adapters, and retraining guidance."""

import unittest

import numpy as np

from src.adaptive_surrogate import (
    AdaptiveBlackBox,
    CandidateResult,
    ResourceBudget,
    available_adapters,
    evaluate_candidates,
    recommend_retraining,
    select_candidate,
)


class LifecycleTest(unittest.TestCase):
    def test_evaluates_and_selects_the_best_feasible_candidate(self) -> None:
        candidates = [
            CandidateResult("fast", 0.2, 1.0, 1.0, 10),
            CandidateResult("accurate", 0.01, 2.0, 1.0, 10),
        ]
        budget = ResourceBudget(max_training_seconds=10.0)

        evaluation = evaluate_candidates(candidates, budget)

        self.assertEqual([item.name for item in evaluation], ["fast", "accurate"])
        self.assertEqual(select_candidate(candidates, budget).name, "accurate")

    def test_treats_zero_budget_as_only_allowing_zero_measured_cost(self) -> None:
        candidate = CandidateResult("free", 0.1, 0.0, 0.0, 0)

        selected = select_candidate([candidate], ResourceBudget(max_training_seconds=0.0))

        self.assertEqual(selected.name, "free")

    def test_reports_adapter_availability_without_importing_optional_packages(self) -> None:
        adapters = {item.name: item.available for item in available_adapters()}

        self.assertTrue(adapters["adaptive_black_box"])
        self.assertIn("autogluon", adapters)
        self.assertIn("pysindy", adapters)

    def test_recommends_offline_retraining_for_significant_drift(self) -> None:
        reference = np.arange(12.0).reshape(-1, 1)
        model = AdaptiveBlackBox(epochs=1).fit(reference, reference, validation_folds=2)

        recommendation = recommend_retraining(model, reference, reference + 100.0)

        self.assertTrue(recommendation.retrain_recommended)
        self.assertIn("distribution shift", recommendation.reasons)
        self.assertIn("does not retrain", recommendation.summary)


if __name__ == "__main__":
    unittest.main()
