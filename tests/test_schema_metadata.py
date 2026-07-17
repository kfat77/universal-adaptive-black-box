import pickle

import numpy as np
import pytest

from adaptive_surrogate import AdaptiveBlackBox


def test_fit_persists_tabular_schema_and_validation_metadata(tmp_path):
    x = np.linspace(-1.0, 1.0, 18).reshape(-1, 1)
    model = AdaptiveBlackBox(epochs=5).fit(
        x,
        x**2,
        feature_names=["temperature"],
        target_names=["response"],
        validation_strategy="holdout",
    )
    path = tmp_path / "model.joblib"
    model.save(path)

    with path.open("rb") as file:
        payload = pickle.load(file)

    assert payload["metadata"]["feature_names"] == ("temperature",)
    assert payload["metadata"]["target_names"] == ("response",)
    assert payload["metadata"]["validation_strategy"] == "holdout"


def test_schema_names_must_match_dimensions():
    x = np.linspace(-1.0, 1.0, 18).reshape(-1, 1)
    with pytest.raises(ValueError, match="feature_names"):
        AdaptiveBlackBox(epochs=5).fit(x, x, feature_names=["a", "b"])
