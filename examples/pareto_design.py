"""Filter predicted multi-objective candidates to a Pareto front."""

import numpy as np

from adaptive_surrogate import AdaptiveBlackBox, non_dominated_mask


def main() -> None:
    x = np.linspace(-1.0, 1.0, 80).reshape(-1, 1)
    y = np.column_stack((1.0 - (x[:, 0] - 0.4) ** 2, (x[:, 0] + 0.2) ** 2))
    engine = AdaptiveBlackBox(epochs=60).fit(x, y)
    candidates = np.linspace(-1.0, 1.0, 101).reshape(-1, 1)
    predicted = engine.predict(candidates)
    front = non_dominated_mask(predicted, ["maximize", "minimize"])
    print("Pareto candidate inputs:", candidates[front, 0])


if __name__ == "__main__":
    main()
