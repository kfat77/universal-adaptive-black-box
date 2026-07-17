"""Artifact schema and backwards-compatibility tests."""

import pickle
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

from src.core_engine import ARTIFACT_VERSION, AdaptiveBlackBox


class ArtifactTest(unittest.TestCase):
    def test_saved_artifact_uses_versioned_schema_and_round_trips(self) -> None:
        X = np.linspace(-1.0, 1.0, 20).reshape(-1, 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            model = AdaptiveBlackBox(epochs=2).fit(X, X, validation_folds=2)
            model.save(path)
            with path.open("rb") as artifact_file:
                payload = pickle.load(artifact_file)
            restored = AdaptiveBlackBox.load(path)

        self.assertEqual(payload["artifact_version"], ARTIFACT_VERSION)
        self.assertEqual(payload["metadata"]["training_samples"], len(X))
        np.testing.assert_allclose(restored.predict([[0.2]]), model.predict([[0.2]]))

    def test_legacy_artifact_loads_with_warning(self) -> None:
        X = np.linspace(-1.0, 1.0, 20).reshape(-1, 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.joblib"
            model = AdaptiveBlackBox(epochs=2).fit(X, X, validation_folds=2)
            legacy = model.__dict__.copy()
            legacy["artifact_version"] = 1
            if model.model_name == "mlp":
                legacy["model"] = None
                legacy["mlp_state"] = model.model.state_dict()
            with path.open("wb") as artifact_file:
                pickle.dump(legacy, artifact_file)
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                restored = AdaptiveBlackBox.load(path)

        self.assertTrue(captured)
        self.assertEqual(restored.input_dim, 1)

    def test_malformed_version_two_artifact_fails_with_schema_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.joblib"
            with path.open("wb") as artifact_file:
                pickle.dump({"artifact_version": 2, "metadata": {}, "state": {}}, artifact_file)
            with self.assertRaisesRegex(ValueError, "required state fields"):
                AdaptiveBlackBox.load(path)
