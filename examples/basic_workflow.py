"""Fit a one-dimensional numerical surrogate and make a forward prediction."""

import numpy as np

from adaptive_surrogate import AdaptiveBlackBox


def main() -> None:
    rng = np.random.default_rng(42)
    x_train = np.linspace(-3.0, 3.0, 120).reshape(-1, 1)
    y_train = np.sin(x_train) + rng.normal(0.0, 0.03, size=x_train.shape)

    engine = AdaptiveBlackBox(epochs=120).fit(x_train, y_train)
    prediction = engine.predict([[1.0]])
    selected_metrics = engine.metrics[engine.model_name]

    print(f"Selected model: {engine.model_name}")
    print(f"Validation nRMSE: {selected_metrics['nrmse']:.4f}")
    print(f"Prediction at x=1.0: {prediction.ravel()[0]:.4f}")


if __name__ == "__main__":
    main()
