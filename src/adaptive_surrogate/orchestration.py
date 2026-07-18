"""Small, explicit orchestration primitives for resource-constrained surrogate tasks."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from .core_engine import AdaptiveBlackBox
from .data_loader import TabularDataset


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


@dataclass(frozen=True)
class DiagnosticReport:
    """Task diagnosis in structured and user-readable forms."""

    profile: TaskProfile
    route: RouteRecommendation
    summary: str


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


def diagnose_dataset(dataset: TabularDataset, spec: TaskSpec) -> DiagnosticReport:
    """Diagnose a loaded dataset when its stored schema matches the task description."""
    if dataset.feature_names != spec.feature_names or dataset.target_names != spec.target_names:
        raise ValueError("TaskSpec names must match the loaded dataset schema.")
    targets = np.asarray(dataset.Y)
    if targets.ndim != 2 or targets.shape[0] != dataset.X.shape[0]:
        raise ValueError("Dataset inputs and targets must be two-dimensional with matching rows.")
    if targets.shape[1] != len(dataset.target_names):
        raise ValueError("Dataset target names must match target columns.")
    profile = profile_task(spec, dataset.X)
    route = route_task(profile)
    summary = (
        f"{profile.task_kind}: {profile.n_samples} samples, {profile.n_features} features, "
        f"and {profile.n_targets} targets. Recommended route: {route.route}."
    )
    return DiagnosticReport(profile, route, summary)


def fit_task(
    engine: AdaptiveBlackBox, dataset: TabularDataset, spec: TaskSpec, **fit_options: Any
) -> AdaptiveBlackBox:
    """Train an existing engine using the validation strategy selected by diagnosis."""
    report = diagnose_dataset(dataset, spec)
    strategy = "time_series" if report.route.route == "time_aware_validation" else "kfold"
    requested_strategy = fit_options.pop("validation_strategy", strategy)
    if requested_strategy != strategy:
        raise ValueError("validation_strategy conflicts with the diagnosed task route.")
    for option, expected in (
        ("feature_names", dataset.feature_names),
        ("target_names", dataset.target_names),
    ):
        supplied = fit_options.pop(option, None)
        if supplied is not None and tuple(supplied) != expected:
            raise ValueError(f"{option} conflicts with the loaded dataset schema.")
    inputs, targets = dataset.X, dataset.Y
    if spec.time_column is not None:
        time_index = dataset.feature_names.index(spec.time_column)
        order = np.argsort(inputs[:, time_index], kind="stable")
        inputs, targets = inputs[order], targets[order]
        if "groups" in fit_options:
            groups = np.asarray(fit_options["groups"])
            if groups.shape != (len(order),):
                raise ValueError("groups must contain one value per dataset row.")
            fit_options["groups"] = groups[order]
    return engine.fit(
        inputs,
        targets,
        validation_strategy=strategy,
        feature_names=dataset.feature_names,
        target_names=dataset.target_names,
        **fit_options,
    )


def score_candidate(candidate: CandidateResult, budget: ResourceBudget) -> CandidateScore:
    """Score feasible candidates by error plus mean normalized resource use."""
    measurements = (
        (candidate.training_seconds, budget.max_training_seconds),
        (candidate.prediction_milliseconds, budget.max_prediction_milliseconds),
        (candidate.model_bytes, budget.max_model_bytes),
    )
    ratios = [
        0.0 if limit == 0 and value == 0 else float("inf") if limit == 0 else value / limit
        for value, limit in measurements
        if limit is not None
    ]
    within_budget = all(ratio <= 1.0 for ratio in ratios)
    value = candidate.error + float(np.mean(ratios)) if within_budget else float("inf")
    return CandidateScore(value, within_budget)
