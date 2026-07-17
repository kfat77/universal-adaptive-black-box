"""Training and persistence for an adaptive numerical black-box model."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import pickle

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch import nn

ARTIFACT_VERSION = 1


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

    def _fit_mlp(self, X: np.ndarray, Y: np.ndarray, seed: int) -> MLP:
        """Fit one standardized MLP candidate and return it in evaluation mode."""
        torch.manual_seed(seed)
        model = MLP(self.input_dim, self.output_dim, self.hidden_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        x_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(Y, dtype=torch.float32)
        model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(x_tensor), y_tensor)
            loss.backward()
            optimizer.step()
        model.eval()
        return model

    def _fit_forest(self, X: np.ndarray, Y: np.ndarray) -> RandomForestRegressor:
        """Fit the statistical candidate, preserving sklearn's single-target API."""
        model = RandomForestRegressor(n_estimators=250, random_state=self.random_state, n_jobs=-1)
        model.fit(X, Y.ravel() if self.output_dim == 1 else Y)
        return model

    def _predict_scaled(self, model: Any, model_name: str, X: np.ndarray) -> np.ndarray:
        if model_name == "mlp":
            with torch.no_grad():
                result = model(torch.tensor(X, dtype=torch.float32)).numpy()
        else:
            result = model.predict(X)
        return np.asarray(result).reshape(len(X), self.output_dim)

    def fit(self, X: np.ndarray, Y: np.ndarray, validation_folds: int = 3) -> "AdaptiveBlackBox":
        """Cross-validate candidates, select by mean MSE, then refit the winner on all data."""
        X, Y = self._as_2d(X, "X"), self._as_2d(Y, "Y")
        if len(X) != len(Y):
            raise ValueError("X and Y must have the same number of rows.")
        if not 2 <= validation_folds <= len(X) // 2:
            raise ValueError("validation_folds must leave at least two samples in every validation fold.")

        self.input_dim, self.output_dim = X.shape[1], Y.shape[1]
        scores = {"mlp": {"mse": [], "r2": []}, "random_forest": {"mse": [], "r2": []}}
        splitter = KFold(n_splits=validation_folds, shuffle=True, random_state=self.random_state)
        for fold, (train_index, validation_index) in enumerate(splitter.split(X)):
            x_scaler, y_scaler = StandardScaler(), StandardScaler()
            X_train = x_scaler.fit_transform(X[train_index])
            Y_train = y_scaler.fit_transform(Y[train_index])
            X_validation, Y_validation = x_scaler.transform(X[validation_index]), Y[validation_index]
            candidates = {
                "mlp": self._fit_mlp(X_train, Y_train, self.random_state + fold),
                "random_forest": self._fit_forest(X_train, Y_train),
            }
            for name, candidate in candidates.items():
                prediction = y_scaler.inverse_transform(self._predict_scaled(candidate, name, X_validation))
                scores[name]["mse"].append(mean_squared_error(Y_validation, prediction))
                scores[name]["r2"].append(r2_score(Y_validation, prediction, multioutput="uniform_average"))
        self.metrics = {
            name: {"mse": float(np.mean(values["mse"])), "mse_std": float(np.std(values["mse"])),
                   "r2": float(np.mean(values["r2"])), "r2_std": float(np.std(values["r2"]))}
            for name, values in scores.items()
        }
        self.model_name = min(self.metrics, key=lambda name: self.metrics[name]["mse"])
        X_full = self.x_scaler.fit_transform(X)
        Y_full = self.y_scaler.fit_transform(Y)
        self.model = (self._fit_mlp(X_full, Y_full, self.random_state)
                      if self.model_name == "mlp" else self._fit_forest(X_full, Y_full))
        return self

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """Run forward solving: map unseen input vectors X directly to outputs Y."""
        if self.model is None or self.input_dim is None:
            raise RuntimeError("Train or load a model before predicting.")
        X_new = self._as_2d(X_new, "X_new")
        if X_new.shape[1] != self.input_dim:
            raise ValueError(f"X_new must contain {self.input_dim} columns.")
        X_scaled = self.x_scaler.transform(X_new)
        prediction_scaled = self._predict_scaled(self.model, self.model_name, X_scaled)
        return self.y_scaler.inverse_transform(prediction_scaled)

    def save(self, path: str | Path) -> None:
        """Persist the selected model, scalers, dimensions, and comparison metrics."""
        if self.model is None:
            raise RuntimeError("Nothing to save: train the model first.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.__dict__.copy()
        payload["artifact_version"] = ARTIFACT_VERSION
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
        if payload.get("artifact_version") != ARTIFACT_VERSION:
            raise ValueError("Unsupported or unversioned model artifact.")
        instance = cls()
        instance.__dict__.update(payload)
        if instance.model_name == "mlp":
            instance.model = MLP(instance.input_dim, instance.output_dim, instance.hidden_dim)
            instance.model.load_state_dict(payload["mlp_state"])
            instance.model.eval()
        return instance
