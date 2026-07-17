"""Training and persistence for an adaptive numerical black-box model."""

from __future__ import annotations

import pickle
import platform
import warnings
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import sklearn
import torch
from scipy.stats import wasserstein_distance
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .metrics import compute_regression_metrics, validate_output_weights
from .validation import build_splits

ARTIFACT_VERSION = 2
PACKAGE_VERSION = "0.2.0"
CANDIDATE_MODELS = (
    "dummy",
    "linear_regression",
    "ridge",
    "mlp",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
)


class MLP(nn.Module):
    """Small fully connected network used for flexible nonlinear fitting."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: tuple[int, ...] = (96, 96),
        dropout: float = 0.0,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.extend([nn.Linear(current_dim, hidden_dim), nn.ReLU()])
            if dropout:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class AdaptiveBlackBox:
    """Train several numerical models and retain the one with lowest validation MSE."""

    def __init__(
        self,
        random_state: int = 42,
        hidden_dim: int = 96,
        epochs: int = 400,
        mlp_config: dict[str, Any] | None = None,
    ):
        self.random_state = random_state
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.mlp_config: dict[str, Any] = {
            "hidden_layers": (hidden_dim, hidden_dim),
            "batch_size": 64,
            "max_epochs": epochs,
            "patience": 30,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "dropout": 0.0,
        } | (mlp_config or {})
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.model: Any | None = None
        self.model_name: str | None = None
        self.metrics: dict[str, dict[str, Any]] = {}
        self.input_dim: int | None = None
        self.output_dim: int | None = None
        self.training_samples: int | None = None
        self.output_weights: np.ndarray | None = None
        self.selection_metric: str = "nrmse"
        self.validation_strategy: str = "kfold"
        self.feature_names: tuple[str, ...] | None = None
        self.target_names: tuple[str, ...] | None = None
        self.calibration_residuals: np.ndarray | None = None
        self.training_x_scaled: np.ndarray | None = None
        self.training_feature_min: np.ndarray | None = None
        self.training_feature_max: np.ndarray | None = None
        self.training_feature_mean: np.ndarray | None = None
        self.training_feature_std: np.ndarray | None = None
        self.ood_distance_threshold: float | None = None

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
        assert self.input_dim is not None and self.output_dim is not None
        config = self.mlp_config
        model = MLP(
            self.input_dim,
            self.output_dim,
            tuple(config["hidden_layers"]),
            float(config["dropout"]),
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
        x_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(Y, dtype=torch.float32)
        validation_size = max(1, int(len(X) * 0.1)) if len(X) >= 10 else 0
        permutation = torch.randperm(len(X), generator=torch.Generator().manual_seed(seed))
        validation_indices, training_indices = (
            permutation[:validation_size],
            permutation[validation_size:],
        )
        training_data = TensorDataset(x_tensor[training_indices], y_tensor[training_indices])
        loader = DataLoader(
            training_data,
            batch_size=min(int(config["batch_size"]), len(training_data)),
            shuffle=True,
        )
        best_state, best_loss, stale_epochs = None, float("inf"), 0
        model.train()
        for _ in range(int(config["max_epochs"])):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                loss = nn.functional.mse_loss(model(batch_x), batch_y)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                check_x, check_y = (
                    (x_tensor[validation_indices], y_tensor[validation_indices])
                    if validation_size
                    else (x_tensor, y_tensor)
                )
                validation_loss = float(nn.functional.mse_loss(model(check_x), check_y))
            model.train()
            if validation_loss < best_loss:
                best_loss, stale_epochs = validation_loss, 0
                best_state = {
                    key: value.detach().clone() for key, value in model.state_dict().items()
                }
            else:
                stale_epochs += 1
                if stale_epochs >= int(config["patience"]):
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        return model

    def _fit_forest(self, X: np.ndarray, Y: np.ndarray) -> RandomForestRegressor:
        """Fit the statistical candidate, preserving sklearn's single-target API."""
        model = RandomForestRegressor(n_estimators=250, random_state=self.random_state, n_jobs=-1)
        model.fit(X, Y.ravel() if self.output_dim == 1 else Y)
        return model

    def _fit_sklearn(self, model: Any, X: np.ndarray, Y: np.ndarray) -> Any:
        return model.fit(X, Y.ravel() if self.output_dim == 1 else Y)

    def _fit_extra_trees(self, X: np.ndarray, Y: np.ndarray) -> ExtraTreesRegressor:
        """Fit a more randomized tree ensemble that can capture fine interactions."""
        model = ExtraTreesRegressor(n_estimators=250, random_state=self.random_state, n_jobs=-1)
        model.fit(X, Y.ravel() if self.output_dim == 1 else Y)
        return model

    def _fit_gradient_boosting(self, X: np.ndarray, Y: np.ndarray) -> Any:
        """Fit a boosted-tree candidate, wrapping it for multi-output targets."""
        base_model = HistGradientBoostingRegressor(max_iter=200, random_state=self.random_state)
        if self.output_dim == 1:
            return base_model.fit(X, Y.ravel())
        return MultiOutputRegressor(base_model).fit(X, Y)

    def _fit_candidate(self, model_name: str, X: np.ndarray, Y: np.ndarray, seed: int) -> Any:
        registry = {
            "dummy": lambda: self._fit_sklearn(DummyRegressor(strategy="mean"), X, Y),
            "linear_regression": lambda: self._fit_sklearn(LinearRegression(), X, Y),
            "ridge": lambda: self._fit_sklearn(
                Ridge(alpha=1.0, random_state=self.random_state), X, Y
            ),
            "mlp": lambda: self._fit_mlp(X, Y, seed),
            "random_forest": lambda: self._fit_forest(X, Y),
            "extra_trees": lambda: self._fit_extra_trees(X, Y),
            "hist_gradient_boosting": lambda: self._fit_gradient_boosting(X, Y),
        }
        try:
            return registry[model_name]()
        except KeyError as error:
            raise ValueError(f"Unknown candidate model: {model_name}") from error

    @staticmethod
    def _model_category(model_name: str) -> str:
        if model_name in {"dummy", "linear_regression", "ridge"}:
            return "baseline"
        return "neural_network" if model_name == "mlp" else "tree_ensemble"

    def _predict_scaled(self, model: Any, model_name: str, X: np.ndarray) -> np.ndarray:
        assert self.output_dim is not None
        if model_name == "mlp":
            with torch.no_grad():
                result = model(torch.tensor(X, dtype=torch.float32)).numpy()
        else:
            result = model.predict(X)
        return np.asarray(result).reshape(len(X), self.output_dim)

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        validation_folds: int = 3,
        output_weights: list[float] | np.ndarray | None = None,
        selection_metric: str = "nrmse",
        validation_strategy: str = "kfold",
        groups: np.ndarray | None = None,
        holdout_fraction: float = 0.2,
        feature_names: list[str] | tuple[str, ...] | None = None,
        target_names: list[str] | tuple[str, ...] | None = None,
    ) -> "AdaptiveBlackBox":
        """Cross-validate candidates, select one, then refit it on all observations.

        Optional feature and target names are persisted with the artifact so callers
        can retain the tabular-data schema used during training.
        """
        inferred_feature_names = self._column_names_from_dataframe(X)
        inferred_target_names = self._column_names_from_dataframe(Y)
        X, Y = self._as_2d(X, "X"), self._as_2d(Y, "Y")
        if len(X) != len(Y):
            raise ValueError("X and Y must have the same number of rows.")

        self.input_dim, self.output_dim = X.shape[1], Y.shape[1]
        if selection_metric not in {"mse", "rmse", "mae", "r2", "nrmse"}:
            raise ValueError("selection_metric must be one of mse, rmse, mae, r2, or nrmse.")
        self.output_weights = validate_output_weights(output_weights, self.output_dim)
        self.selection_metric = selection_metric
        self.validation_strategy = validation_strategy
        self.feature_names = self._validate_names(
            feature_names if feature_names is not None else inferred_feature_names,
            self.input_dim,
            "feature_names",
        )
        self.target_names = self._validate_names(
            target_names if target_names is not None else inferred_target_names,
            self.output_dim,
            "target_names",
        )
        scores: dict[str, list[dict[str, Any]]] = {name: [] for name in CANDIDATE_MODELS}
        residuals: dict[str, list[np.ndarray]] = {name: [] for name in CANDIDATE_MODELS}
        target_scales = np.ptp(Y, axis=0)
        splits = build_splits(
            validation_strategy,
            len(X),
            validation_folds,
            self.random_state,
            groups,
            holdout_fraction,
        )
        for fold, (train_index, validation_index) in enumerate(splits):
            x_scaler, y_scaler = StandardScaler(), StandardScaler()
            X_train = x_scaler.fit_transform(X[train_index])
            Y_train = y_scaler.fit_transform(Y[train_index])
            X_validation, Y_validation = (
                x_scaler.transform(X[validation_index]),
                Y[validation_index],
            )
            candidates = {}
            for name in CANDIDATE_MODELS:
                started = perf_counter()
                candidates[name] = (
                    self._fit_candidate(name, X_train, Y_train, self.random_state + fold),
                    perf_counter() - started,
                )
            for name, (candidate, training_seconds) in candidates.items():
                prediction_started = perf_counter()
                prediction = y_scaler.inverse_transform(
                    self._predict_scaled(candidate, name, X_validation)
                )
                fold_metrics = compute_regression_metrics(
                    Y_validation,
                    prediction,
                    self.output_weights,
                    normalization_scales=target_scales,
                )
                fold_metrics["training_seconds"] = training_seconds
                fold_metrics["inference_seconds"] = perf_counter() - prediction_started
                scores[name].append(fold_metrics)
                residuals[name].append(np.abs(prediction - Y_validation))
        self.metrics = {
            name: {
                **{
                    metric: float(np.mean([fold[metric] for fold in values]))
                    for metric in ("mse", "rmse", "mae", "r2", "nrmse")
                },
                **{
                    f"{metric}_std": float(np.std([fold[metric] for fold in values]))
                    for metric in (
                        "mse",
                        "rmse",
                        "mae",
                        "r2",
                        "nrmse",
                        "training_seconds",
                        "inference_seconds",
                    )
                },
                **{
                    metric: float(np.mean([fold[metric] for fold in values]))
                    for metric in ("training_seconds", "inference_seconds")
                },
                "fold_metrics": values,
                "baseline": self._model_category(name) == "baseline",
                "model_category": self._model_category(name),
            }
            for name, values in scores.items()
        }
        self.model_name = (
            max(self.metrics, key=lambda name: self.metrics[name]["r2"])
            if selection_metric == "r2"
            else min(self.metrics, key=lambda name: self.metrics[name][selection_metric])
        )
        self.metrics[self.model_name]["selected"] = True
        X_full = self.x_scaler.fit_transform(X)
        Y_full = self.y_scaler.fit_transform(Y)
        self.model = self._fit_candidate(self.model_name, X_full, Y_full, self.random_state)
        self.training_samples = len(X)
        self.calibration_residuals = np.vstack(residuals[self.model_name])
        self.training_x_scaled = X_full
        self.training_feature_min = X.min(axis=0)
        self.training_feature_max = X.max(axis=0)
        self.training_feature_mean = X.mean(axis=0)
        self.training_feature_std = X.std(axis=0)
        distances = np.linalg.norm(X_full[:, None, :] - X_full[None, :, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        self.ood_distance_threshold = float(np.quantile(np.min(distances, axis=1), 0.95))
        return self

    @staticmethod
    def _column_names_from_dataframe(values: Any) -> tuple[str, ...] | None:
        """Return DataFrame column names without making pandas a hard dependency."""
        columns = getattr(values, "columns", None)
        if columns is None:
            return None
        return tuple(columns.tolist())

    @staticmethod
    def _validate_names(
        names: list[str] | tuple[str, ...] | None, expected_count: int, parameter_name: str
    ) -> tuple[str, ...] | None:
        if names is None:
            return None
        normalized = tuple(names)
        if (
            len(normalized) != expected_count
            or len(set(normalized)) != expected_count
            or any(not isinstance(name, str) or not name.strip() for name in normalized)
        ):
            raise ValueError(
                f"{parameter_name} must contain {expected_count} unique, non-empty strings."
            )
        return normalized

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """Run forward solving: map unseen input vectors X directly to outputs Y."""
        if self.model is None or self.input_dim is None:
            raise RuntimeError("Train or load a model before predicting.")
        incoming_names = self._column_names_from_dataframe(X_new)
        if incoming_names is not None and self.feature_names is not None:
            missing = sorted(set(self.feature_names) - set(incoming_names))
            extra = sorted(set(incoming_names) - set(self.feature_names))
            if missing or extra:
                raise ValueError(
                    f"DataFrame prediction columns must match training features; missing={missing}, extra={extra}."
                )
            dataframe: Any = X_new
            X_new = dataframe.loc[:, list(self.feature_names)]
        X_new = self._as_2d(X_new, "X_new")
        if X_new.shape[1] != self.input_dim:
            raise ValueError(f"X_new must contain {self.input_dim} columns.")
        X_scaled = self.x_scaler.transform(X_new)
        assert self.model_name is not None
        prediction_scaled = self._predict_scaled(self.model, self.model_name, X_scaled)
        return self.y_scaler.inverse_transform(prediction_scaled)

    def predict_interval(
        self, X_new: np.ndarray, confidence: float = 0.9
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return prediction intervals calibrated from cross-validation residuals.

        The interval assumes calibration and future samples are exchangeable; it is
        not a physical, causal, or distribution-shift guarantee.
        """
        if not 0 < confidence < 1 or self.calibration_residuals is None:
            raise ValueError("confidence must be between 0 and 1 after fitting the model.")
        prediction = self.predict(X_new)
        radius = np.quantile(self.calibration_residuals, confidence, axis=0)
        return prediction, prediction - radius, prediction + radius

    def assess_distribution(self, X_new: np.ndarray) -> dict[str, np.ndarray]:
        """Assess nearest-neighbour distance and feature-range extrapolation risk."""
        if self.training_x_scaled is None or self.ood_distance_threshold is None:
            raise RuntimeError("Train or load a model before assessing distribution.")
        values = self._as_2d(X_new, "X_new")
        if values.shape[1] != self.input_dim:
            raise ValueError(f"X_new must contain {self.input_dim} columns.")
        scaled = self.x_scaler.transform(values)
        nearest = np.min(
            np.linalg.norm(scaled[:, None, :] - self.training_x_scaled[None, :, :], axis=2), axis=1
        )
        outside = (values < self.training_feature_min) | (values > self.training_feature_max)
        in_distribution = (nearest <= self.ood_distance_threshold) & ~outside.any(axis=1)
        score = nearest / max(self.ood_distance_threshold, np.finfo(float).eps)
        risk_level = np.where(score > 2.0, "high", np.where(score > 1.0, "medium", "low"))
        explanation = np.array(
            [
                "one or more features are outside the training range"
                if row_outside.any()
                else "nearest-neighbour distance is above the training threshold"
                if row_score > 1.0
                else "within feature ranges and nearest-neighbour threshold"
                for row_outside, row_score in zip(outside, score, strict=True)
            ],
            dtype=object,
        )
        return {
            "nearest_training_distance": nearest,
            "features_outside_training_range": outside,
            "extrapolation_score": score,
            "in_distribution": in_distribution,
            "risk_level": risk_level,
            "explanation": explanation,
        }

    def compare_data_distribution(
        self, X_reference: np.ndarray, X_new: np.ndarray
    ) -> dict[str, Any]:
        """Compare feature distributions; this reports drift and never adapts the model.

        Inputs must preserve the same feature order used for training. Mean shifts are
        reported in reference-standard-deviation units and Wasserstein distances are
        reported in original feature units.
        """
        reference, new = self._as_2d(X_reference, "X_reference"), self._as_2d(X_new, "X_new")
        if (
            self.input_dim is None
            or reference.shape[1] != self.input_dim
            or new.shape[1] != self.input_dim
        ):
            raise ValueError(f"Both datasets must contain {self.input_dim} feature columns.")
        reference_mean = reference.mean(axis=0)
        reference_std = reference.std(axis=0)
        stable_std = np.where(reference_std > np.finfo(float).eps, reference_std, 1.0)
        assessment = self.assess_distribution(new)
        return {
            "feature_mean_shift": (new.mean(axis=0) - reference_mean) / stable_std,
            "feature_std_ratio": new.std(axis=0) / stable_std,
            "feature_wasserstein_distance": np.array(
                [
                    wasserstein_distance(reference[:, column], new[:, column])
                    for column in range(self.input_dim)
                ]
            ),
            "feature_range_shift": (new.min(axis=0) < reference.min(axis=0))
            | (new.max(axis=0) > reference.max(axis=0)),
            "ood_rate": float(1.0 - np.mean(assessment["in_distribution"])),
            "note": "This report detects distribution change; it does not update or adapt the model.",
        }

    def refit(self, X: np.ndarray, Y: np.ndarray, **fit_options: Any) -> "AdaptiveBlackBox":
        """Explicitly retrain on updated data; this is not online learning."""
        return self.fit(X, Y, **fit_options)

    def recommend_next_experiments(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        """Recommend candidate experiments without executing them in the real world."""
        from .active_learning import recommend_next_experiments

        return recommend_next_experiments(self, *args, **kwargs)

    def save(self, path: str | Path) -> None:
        """Persist the selected model, scalers, dimensions, and comparison metrics."""
        if self.model is None:
            raise RuntimeError("Nothing to save: train the model first.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = self.__dict__.copy()
        if self.model_name == "mlp":
            state["model"] = None
            state["mlp_state"] = self.model.state_dict()
        payload = {
            "artifact_version": ARTIFACT_VERSION,
            "metadata": {
                "package_version": PACKAGE_VERSION,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "python_version": platform.python_version(),
                "numpy_version": np.__version__,
                "scikit_learn_version": sklearn.__version__,
                "torch_version": torch.__version__,
                "model_name": self.model_name,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "training_samples": self.training_samples,
                "selection_metric": self.selection_metric,
                "validation_strategy": self.validation_strategy,
                "output_weights": None
                if self.output_weights is None
                else self.output_weights.tolist(),
                "feature_names": self.feature_names,
                "target_names": self.target_names,
            },
            "state": state,
        }
        with path.open("wb") as artifact_file:
            pickle.dump(payload, artifact_file)

    @classmethod
    def load(cls, path: str | Path) -> "AdaptiveBlackBox":
        """Restore a model saved by :meth:`save`."""
        with Path(path).open("rb") as artifact_file:
            payload = pickle.load(artifact_file)
        version = payload.get("artifact_version")
        if version == 1:
            warnings.warn(
                "Loading legacy artifact version 1; re-save it to upgrade the schema.", UserWarning
            )
            state = payload
        elif version == ARTIFACT_VERSION:
            state = payload.get("state")
            if not isinstance(state, dict):
                raise ValueError("Artifact schema is missing model state.")
            required = {
                "model_name",
                "input_dim",
                "output_dim",
                "x_scaler",
                "y_scaler",
                "output_weights",
            }
            missing = sorted(required - state.keys())
            if missing:
                raise ValueError(f"Artifact schema is missing required state fields: {missing}")
        else:
            raise ValueError("Unsupported or unversioned model artifact.")
        instance = cls()
        instance.__dict__.update(state)
        if not hasattr(instance, "mlp_config"):
            instance.mlp_config = {
                "hidden_layers": (instance.hidden_dim, instance.hidden_dim),
                "dropout": 0.0,
            }
        if instance.model_name == "mlp":
            if "mlp_state" not in state:
                raise ValueError("MLP artifact schema is missing mlp_state.")
            assert instance.input_dim is not None and instance.output_dim is not None
            instance.model = MLP(
                instance.input_dim,
                instance.output_dim,
                tuple(instance.mlp_config["hidden_layers"]),
                float(instance.mlp_config.get("dropout", 0.0)),
            )
            instance.model.load_state_dict(state["mlp_state"])
            instance.model.eval()
        return instance
