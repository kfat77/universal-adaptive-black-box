"""Training and persistence for an adaptive numerical black-box model."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import pickle

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn


class MLP(nn.Module):
    """Small fully connected network used for flexible nonlinear fitting."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 96):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class AdaptiveBlackBox:
    """Train several numerical models and retain the one with lowest validation MSE."""

    def __init__(self, random_state: int = 42, hidden_dim: int = 96, epochs: int = 400):
        self.random_state = random_state
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.model: Any | None = None
        self.model_name: str | None = None
        self.metrics: dict[str, dict[str, float]] = {}
        self.input_dim: int | None = None
        self.output_dim: int | None = None

    @staticmethod
    def _as_2d(values: np.ndarray, name: str) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 1:
            array = array[:, None]
        if array.ndim != 2 or len(array) == 0 or not np.isfinite(array).all():
            raise ValueError(f"{name} must be a non-empty, finite 2D numerical array.")
        return array

    def fit(self, X: np.ndarray, Y: np.ndarray, validation_fraction: float = 0.2) -> "AdaptiveBlackBox":
        """Split data, fit MLP and statistical regression, then choose the best validator."""
        X, Y = self._as_2d(X, "X"), self._as_2d(Y, "Y")
        if len(X) != len(Y):
            raise ValueError("X and Y must have the same number of rows.")
        if not 0.0 < validation_fraction < 0.5:
            raise ValueError("validation_fraction must be between 0 and 0.5.")

        self.input_dim, self.output_dim = X.shape[1], Y.shape[1]
        X_train, X_val, Y_train, Y_val = train_test_split(
            X, Y, test_size=validation_fraction, random_state=self.random_state
        )
        X_train_s = self.x_scaler.fit_transform(X_train)
        X_val_s = self.x_scaler.transform(X_val)
        Y_train_s = self.y_scaler.fit_transform(Y_train)

        # Candidate 1: neural nonlinear regression trained in standardized space.
        torch.manual_seed(self.random_state)
        mlp = MLP(self.input_dim, self.output_dim, self.hidden_dim)
        optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()
        x_tensor = torch.tensor(X_train_s, dtype=torch.float32)
        y_tensor = torch.tensor(Y_train_s, dtype=torch.float32)
        mlp.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            loss = loss_fn(mlp(x_tensor), y_tensor)
            loss.backward()
            optimizer.step()
        mlp.eval()
        with torch.no_grad():
            mlp_prediction = self.y_scaler.inverse_transform(
                mlp(torch.tensor(X_val_s, dtype=torch.float32)).numpy()
            )

        # Candidate 2: a tree ensemble, often resilient to noisy numerical data.
        forest = RandomForestRegressor(n_estimators=250, random_state=self.random_state, n_jobs=-1)
        # sklearn treats a single target specially and returns a one-dimensional result.
        # Keep that API happy while restoring a two-dimensional matrix before scaling.
        forest_target = Y_train_s.ravel() if self.output_dim == 1 else Y_train_s
        forest.fit(X_train_s, forest_target)
        forest_prediction = self.y_scaler.inverse_transform(
            np.asarray(forest.predict(X_val_s)).reshape(len(X_val_s), self.output_dim)
        )

        candidates = {"mlp": (mlp, mlp_prediction), "random_forest": (forest, forest_prediction)}
        for name, (_, prediction) in candidates.items():
            self.metrics[name] = {
                "mse": float(mean_squared_error(Y_val, prediction)),
                "r2": float(r2_score(Y_val, prediction, multioutput="uniform_average")),
            }
        self.model_name = min(self.metrics, key=lambda name: self.metrics[name]["mse"])
        self.model = candidates[self.model_name][0]
        return self

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """Run forward solving: map unseen input vectors X directly to outputs Y."""
        if self.model is None or self.input_dim is None:
            raise RuntimeError("Train or load a model before predicting.")
        X_new = self._as_2d(X_new, "X_new")
        if X_new.shape[1] != self.input_dim:
            raise ValueError(f"X_new must contain {self.input_dim} columns.")
        X_scaled = self.x_scaler.transform(X_new)
        if self.model_name == "mlp":
            self.model.eval()
            with torch.no_grad():
                prediction_scaled = self.model(torch.tensor(X_scaled, dtype=torch.float32)).numpy()
        else:
            prediction_scaled = np.asarray(self.model.predict(X_scaled)).reshape(len(X_new), self.output_dim)
        return self.y_scaler.inverse_transform(np.asarray(prediction_scaled).reshape(len(X_new), -1))

    def save(self, path: str | Path) -> None:
        """Persist the selected model, scalers, dimensions, and comparison metrics."""
        if self.model is None:
            raise RuntimeError("Nothing to save: train the model first.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.__dict__.copy()
        if self.model_name == "mlp":
            payload["model"] = None
            payload["mlp_state"] = self.model.state_dict()
        with path.open("wb") as artifact_file:
            pickle.dump(payload, artifact_file)

    @classmethod
    def load(cls, path: str | Path) -> "AdaptiveBlackBox":
        """Restore a model saved by :meth:`save`."""
        with Path(path).open("rb") as artifact_file:
            payload = pickle.load(artifact_file)
        instance = cls()
        instance.__dict__.update(payload)
        if instance.model_name == "mlp":
            instance.model = MLP(instance.input_dim, instance.output_dim, instance.hidden_dim)
            instance.model.load_state_dict(payload["mlp_state"])
            instance.model.eval()
        return instance
