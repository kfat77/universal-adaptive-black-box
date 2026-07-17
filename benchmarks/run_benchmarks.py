"""Run small, deterministic numerical surrogate smoke benchmarks."""

import numpy as np

from adaptive_surrogate import AdaptiveBlackBox


def make_sine_data() -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-3.0, 3.0, 120).reshape(-1, 1)
    return x, np.sin(x)


def make_interaction_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(11)
    x = rng.uniform(-1.0, 1.0, size=(150, 2))
    return x, (x[:, [0]] * x[:, [1]] + 0.2 * x[:, [0]] ** 2)


def main() -> None:
    for name, factory in {"sine": make_sine_data, "interaction": make_interaction_data}.items():
        x, y = factory()
        engine = AdaptiveBlackBox(epochs=80).fit(x, y)
        selected = engine.metrics[engine.model_name]
        dummy = engine.metrics["dummy"]
        print(
            f"{name}: selected={engine.model_name}, nRMSE={selected['nrmse']:.4f}, "
            f"dummy_nRMSE={dummy['nrmse']:.4f}"
        )


if __name__ == "__main__":
    main()
