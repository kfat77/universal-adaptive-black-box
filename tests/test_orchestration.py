"""Public-interface tests for task diagnosis, routing, and candidate scoring."""

import unittest

import numpy as np

from src.adaptive_surrogate import (
    CandidateResult,
    ResourceBudget,
    TaskSpec,
    profile_task,
    route_task,
    score_candidate,
)


class OrchestrationTest(unittest.TestCase):
    def test_profiles_a_numerical_tabular_task(self) -> None:
        spec = TaskSpec(feature_names=("temperature", "pressure"), target_names=("yield",))

        profile = profile_task(spec, np.ones((12, 2)))

        self.assertEqual(profile.task_kind, "tabular_regression")
        self.assertEqual(profile.n_features, 2)
        self.assertEqual(profile.n_samples, 12)

    def test_routes_time_ordered_data_to_time_series_path(self) -> None:
        spec = TaskSpec(
            feature_names=("timestamp", "temperature"),
            target_names=("rainfall",),
            time_column="timestamp",
        )

        route = route_task(profile_task(spec, np.ones((20, 2))))

        self.assertEqual(route.route, "time_aware_validation")
        self.assertIn("time", route.reason)

    def test_scores_only_candidates_within_budget(self) -> None:
        budget = ResourceBudget(
            max_training_seconds=5.0, max_prediction_milliseconds=2.0, max_model_bytes=100
        )
        candidate = CandidateResult(
            name="small-model",
            error=0.1,
            training_seconds=2.0,
            prediction_milliseconds=1.0,
            model_bytes=80,
        )

        score = score_candidate(candidate, budget)

        self.assertTrue(score.within_budget)
        self.assertAlmostEqual(score.value, 0.1 + (0.4 + 0.5 + 0.8) / 3)

    def test_rejects_candidates_outside_budget(self) -> None:
        candidate = CandidateResult(
            name="large-model",
            error=0.01,
            training_seconds=6.0,
            prediction_milliseconds=1.0,
            model_bytes=80,
        )

        score = score_candidate(candidate, ResourceBudget(max_training_seconds=5.0))

        self.assertFalse(score.within_budget)
        self.assertEqual(score.value, float("inf"))

    def test_rejects_invalid_schema_and_resource_values(self) -> None:
        with self.assertRaises(ValueError):
            TaskSpec(feature_names=("x",), target_names=("y",), time_column="timestamp")
        with self.assertRaises(ValueError):
            ResourceBudget(max_training_seconds=-1.0)
        with self.assertRaises(ValueError):
            CandidateResult("bad", float("nan"), 1.0, 1.0, 1)


if __name__ == "__main__":
    unittest.main()
