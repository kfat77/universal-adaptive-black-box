"""Small deterministic benchmark datasets that never require network access."""

import numpy as np
from sklearn.datasets import load_diabetes, make_friedman1


def benchmark_datasets() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return local regression datasets suitable for a quick CPU benchmark."""
    friedman_x, friedman_y = make_friedman1(n_samples=120, n_features=5, noise=0.5, random_state=7)
    diabetes = load_diabetes()
    linear_x = np.linspace(-2.0, 2.0, 120).reshape(-1, 1)
    nonlinear_x = np.linspace(-3.0, 3.0, 120).reshape(-1, 1)
    multi_x = np.linspace(-2.0, 2.0, 120).reshape(-1, 1)
    return {
        "friedman1": (friedman_x, friedman_y[:, None]),
        "diabetes": (diabetes.data, diabetes.target[:, None]),
        "synthetic_linear": (linear_x, 2.0 * linear_x - 0.5),
        "synthetic_nonlinear": (nonlinear_x, np.sin(nonlinear_x)),
        "synthetic_multi_output": (
            multi_x,
            np.column_stack((np.sin(multi_x[:, 0]), multi_x[:, 0] ** 2)),
        ),
    }
