"""Numerical surrogate modeling, uncertainty assessment, and inverse design."""

from .active_learning import recommend_next_experiments
from .core_engine import AdaptiveBlackBox
from .data_loader import TabularDataset, load_tabular_data
from .explainability import local_sensitivity, permutation_importance
from .forward_solver import ForwardSolver
from .inverse_solver import InverseSolver
from .orchestration import (
    CandidateResult,
    CandidateScore,
    ResourceBudget,
    RouteRecommendation,
    TaskProfile,
    TaskSpec,
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
    "TabularDataset",
    "load_tabular_data",
    "local_sensitivity",
    "permutation_importance",
    "recommend_next_experiments",
    "non_dominated_mask",
    "CandidateResult",
    "CandidateScore",
    "ResourceBudget",
    "RouteRecommendation",
    "TaskProfile",
    "TaskSpec",
    "profile_task",
    "route_task",
    "score_candidate",
]
