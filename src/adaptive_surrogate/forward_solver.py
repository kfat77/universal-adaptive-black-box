"""Convenience interface for loading an artifact and making forward predictions."""

import numpy as np

from .core_engine import AdaptiveBlackBox


class ForwardSolver:
    def __init__(self, artifact_path: str):
        self.model = AdaptiveBlackBox.load(artifact_path)

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """Evaluate the learned forward black-box mapping X -> Y."""
        return self.model.predict(X_new)
