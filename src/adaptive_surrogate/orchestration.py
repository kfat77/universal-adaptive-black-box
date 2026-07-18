"""Small, explicit orchestration primitives for resource-constrained surrogate tasks."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TaskSpec:
    """User-supplied schema for a numerical surrogate task."""

    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]
    time_column: str | None = None

    def __post_init__(self) -> None:
        if not self.feature_names or not self.target_names:
            raise ValueError("TaskSpec requires at least one feature and target.")
        if len(set(self.feature_names + self.target_names)) != len(
            self.feature_names + self.target_names
        ):
            raise ValueError("Feature and target names must be unique.")
        if self.time_column is not None and self.time_column not in self.feature_names:
            raise ValueError("time_column must be one of the feature names.")


@dataclass(frozen=True)
class TaskProfile:
    task_kind: str
    n_samples: int
    n_features: int
    n_targets: int


@dataclass(frozen=True)
class RouteRecommendation:
    route: str
    reason: str


@dataclass(frozen=True)
class ResourceBudget:
    max_training_seconds: float | None = None
    max_prediction_milliseconds: float | None = None
    max_model_bytes: int | None = None

    def __post_init__(self) -> None:
        limits = (
            self.max_training_seconds,
            self.max_prediction_milliseconds,
            self.max_model_bytes,
        )
        if any(limit is not None and (not np.isfinite(limit) or limit < 0) for limit in limits):
            raise ValueError("Resource limits must be finite, non-negative values.")


@dataclass(frozen=True)
class CandidateResult:
    name: str
    error: float
    training_seconds: float
    prediction_milliseconds: float
    model_bytes: int

    def __post_init__(self) -> None:
        values = (self.error, self.training_seconds, self.prediction_milliseconds, self.model_bytes)
        if any(not np.isfinite(value) or value < 0 for value in values):
            raise ValueError("Candidate measurements must be finite, non-negative values.")


@dataclass(frozen=True)
class CandidateScore:
    value: float
    within_budget: bool


def profile_task(spec: TaskSpec, features: np.ndarray) -> TaskProfile:
    """Summarize a validated numerical feature matrix for routing."""
    matrix = np.asarray(features)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("features must be a non-empty two-dimensional array.")
    if matrix.shape[1] != len(spec.feature_names):
        raise ValueError("features must match TaskSpec.feature_names.")
    return TaskProfile(
        task_kind="time_series" if spec.time_column else "tabular_regression",
        n_samples=matrix.shape[0],
        n_features=matrix.shape[1],
        n_targets=len(spec.target_names),
    )


def route_task(profile: TaskProfile) -> RouteRecommendation:
    """Choose the first-version modeling route from a task profile."""
    if profile.task_kind == "time_series":
        return RouteRecommendation(
            "time_aware_validation", "A time column requires order-aware validation."
        )
    return RouteRecommendation(
        "tabular_regression", "Numerical features have no declared time order."
    )


def score_candidate(candidate: CandidateResult, budget: ResourceBudget) -> CandidateScore:
    """Score feasible candidates by error plus mean normalized resource use."""
    measurements = (
        (candidate.training_seconds, budget.max_training_seconds),
        (candidate.prediction_milliseconds, budget.max_prediction_milliseconds),
        (candidate.model_bytes, budget.max_model_bytes),
    )
    ratios = [value / limit for value, limit in measurements if limit is not None]
    within_budget = all(ratio <= 1.0 for ratio in ratios)
    value = candidate.error + float(np.mean(ratios)) if within_budget else float("inf")
    return CandidateScore(value, within_budget)
