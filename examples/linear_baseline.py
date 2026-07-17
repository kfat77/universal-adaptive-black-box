"""Inspect linear and Dummy baselines alongside the selected surrogate."""

import numpy as np

from adaptive_surrogate import AdaptiveBlackBox


def main() -> None:
    x = np.linspace(-2.0, 2.0, 80).reshape(-1, 1)
    y = 1.5 * x - 0.25
    engine = AdaptiveBlackBox(epochs=20).fit(x, y)
    for name in ("dummy", "linear_regression", "ridge", engine.model_name):
        print(f"{name}: nRMSE={engine.metrics[name]['nrmse']:.6f}")


if __name__ == "__main__":
    main()
