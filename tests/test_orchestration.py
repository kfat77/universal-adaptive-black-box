"""Public-interface tests for task diagnosis, routing, and candidate scoring."""

import unittest

import numpy as np

from src.adaptive_surrogate import (
    CandidateResult,
    ResourceBudget,
    TabularDataset,
    TaskSpec,
    diagnose_dataset,
    profile_task,
    route_task,
    score_candidate,
)


class OrchestrationTest(unittest.TestCase):
    def test_diagnoses_a_loaded_dataset_with_a_user_readable_summary(self) -> None:
        dataset = TabularDataset(
            X=np.ones((12, 2)),
            Y=np.ones((12, 1)),
            feature_names=("temperature", "pressure"),
            target_names=("yield",),
        )
        spec = TaskSpec(feature_names=dataset.feature_names, target_names=dataset.target_names)

        report = diagnose_dataset(dataset, spec)

        self.assertEqual(report.profile.n_samples, 12)
        self.assertEqual(report.route.route, "tabular_regression")
        self.assertIn("12 samples", report.summary)

    def test_rejects_dataset_schema_or_shape_mismatch(self) -> None:
        dataset = TabularDataset(
            X=np.ones((2, 1)),
            Y=np.ones((1, 1)),
            feature_names=("x",),
            target_names=("y",),
        )

        with self.assertRaises(ValueError):
            diagnose_dataset(dataset, TaskSpec(feature_names=("x",), target_names=("y",)))
        with self.assertRaises(ValueError):
            diagnose_dataset(
                TabularDataset(
                    X=np.ones((2, 1)),
                    Y=np.ones((2, 1)),
                    feature_names=("x",),
                    target_names=("y",),
                ),
                TaskSpec(feature_names=("x",), target_names=("other",)),
            )

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
