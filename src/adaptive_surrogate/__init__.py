"""Numerical surrogate modeling, uncertainty assessment, and inverse design."""

from .active_learning import recommend_next_experiments
from .core_engine import AdaptiveBlackBox
from .data_loader import TabularDataset, load_tabular_data
from .explainability import local_sensitivity, permutation_importance
from .forward_solver import ForwardSolver
from .inverse_solver import InverseSolver
from .lifecycle import (
    AdapterStatus,
    EvaluatedCandidate,
    RetrainingRecommendation,
    available_adapters,
    evaluate_candidates,
    recommend_retraining,
    select_candidate,
)
from .orchestration import (
    CandidateResult,
    CandidateScore,
    DiagnosticReport,
    ResourceBudget,
    RouteRecommendation,
    TaskProfile,
    TaskSpec,
    diagnose_dataset,
    fit_task,
    profile_task,
    route_task,
    score_candidate,
)
from .pareto import non_dominated_mask

__version__ = "0.3.1"

__all__ = [
    "AdaptiveBlackBox",
    "ForwardSolver",
    "InverseSolver",
    "AdapterStatus",
    "EvaluatedCandidate",
    "RetrainingRecommendation",
    "TabularDataset",
    "load_tabular_data",
    "local_sensitivity",
    "permutation_importance",
    "recommend_next_experiments",
    "available_adapters",
    "evaluate_candidates",
    "recommend_retraining",
    "select_candidate",
    "non_dominated_mask",
    "CandidateResult",
    "CandidateScore",
    "DiagnosticReport",
    "ResourceBudget",
    "RouteRecommendation",
    "TaskProfile",
    "TaskSpec",
    "diagnose_dataset",
    "fit_task",
    "profile_task",
    "route_task",
    "score_candidate",
]
