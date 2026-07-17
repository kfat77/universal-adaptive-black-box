"""Fit a multi-output numerical surrogate with named target columns."""

import numpy as np

from adaptive_surrogate import AdaptiveBlackBox


def main() -> None:
    x = np.linspace(-2.0, 2.0, 100).reshape(-1, 1)
    y = np.column_stack((np.sin(x[:, 0]), x[:, 0] ** 2))
    engine = AdaptiveBlackBox(epochs=80).fit(
        x,
        y,
        target_names=["signal", "energy"],
        output_weights=[0.6, 0.4],
    )
    print("Selected model:", engine.model_name)
    print("Prediction:", engine.predict([[0.5]]))


if __name__ == "__main__":
    main()
