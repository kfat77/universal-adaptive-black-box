"""Numerical surrogate modeling, uncertainty assessment, and inverse design."""

from .active_learning import recommend_next_experiments
from .core_engine import AdaptiveBlackBox
from .data_loader import TabularDataset, load_tabular_data
from .forward_solver import ForwardSolver
from .inverse_solver import InverseSolver
from .pareto import non_dominated_mask

__version__ = "0.2.0"

__all__ = [
    "AdaptiveBlackBox",
    "ForwardSolver",
    "InverseSolver",
    "TabularDataset",
    "load_tabular_data",
    "recommend_next_experiments",
    "non_dominated_mask",
]
