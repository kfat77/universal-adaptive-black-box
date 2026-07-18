"""Evaluation, optional-adapter discovery, and offline retraining guidance."""

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Sequence

import numpy as np

from .core_engine import AdaptiveBlackBox
from .orchestration import CandidateResult, CandidateScore, ResourceBudget, score_candidate


@dataclass(frozen=True)
class EvaluatedCandidate:
    name: str
    candidate: CandidateResult
    score: CandidateScore


@dataclass(frozen=True)
class AdapterStatus:
    name: str
    available: bool
    package: str | None


@dataclass(frozen=True)
class RetrainingRecommendation:
    retrain_recommended: bool
    reasons: tuple[str, ...]
    summary: str
    drift_report: dict[str, object]


def evaluate_candidates(
    candidates: Sequence[CandidateResult], budget: ResourceBudget
) -> tuple[EvaluatedCandidate, ...]:
    """Score candidates consistently and return the original comparison order."""
    if not candidates:
        raise ValueError("At least one candidate is required.")
    return tuple(
        EvaluatedCandidate(candidate.name, candidate, score_candidate(candidate, budget))
        for candidate in candidates
    )


def select_candidate(
    candidates: Sequence[CandidateResult], budget: ResourceBudget
) -> CandidateResult:
    """Select the lowest-scoring candidate that satisfies every requested budget."""
    feasible = [
        item for item in evaluate_candidates(candidates, budget) if item.score.within_budget
    ]
    if not feasible:
        raise ValueError("No candidate satisfies the resource budget.")
    return min(feasible, key=lambda item: item.score.value).candidate


def available_adapters() -> tuple[AdapterStatus, ...]:
    """Report supported integrations without importing optional packages."""
    return (
        AdapterStatus("adaptive_black_box", True, None),
        AdapterStatus("autogluon", find_spec("autogluon") is not None, "autogluon"),
        AdapterStatus("pysindy", find_spec("pysindy") is not None, "pysindy"),
    )


def recommend_retraining(
    engine: AdaptiveBlackBox,
    reference_features: np.ndarray,
    new_features: np.ndarray,
    mean_shift_threshold: float = 2.0,
    ood_rate_threshold: float = 0.2,
) -> RetrainingRecommendation:
    """Recommend explicit offline retraining when monitored data substantially drifts."""
    if mean_shift_threshold < 0 or not 0 <= ood_rate_threshold <= 1:
        raise ValueError("Drift thresholds must be non-negative, with OOD rate at most one.")
    report = engine.compare_data_distribution(reference_features, new_features)
    shifted = bool(np.max(np.abs(report["feature_mean_shift"])) >= mean_shift_threshold)
    ood = bool(report["ood_rate"] >= ood_rate_threshold)
    reasons = tuple(
        reason
        for condition, reason in ((shifted, "distribution shift"), (ood, "elevated OOD rate"))
        if condition
    )
    retrain = bool(reasons)
    summary = (
        "Offline retraining is recommended; this function does not retrain the model."
        if retrain
        else "No offline retraining is recommended; this function does not retrain the model."
    )
    return RetrainingRecommendation(retrain, reasons, summary, report)
