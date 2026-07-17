# Universal Adaptive Black-Box Engine

Train a numerical input-to-output model directly from data, then use the same saved artifact for forward prediction and inverse solving.

The project is intended for problems where you have measured numerical samples but no reliable physical formula. It compares a neural-network model with a statistical tree-ensemble model, selects the candidate with the lowest validation mean-squared error (MSE), and saves the winner.

## What it does

- Accepts multi-dimensional numerical inputs `X` with shape `(N, D_in)` and targets `Y` with shape `(N, D_out)`.
- Uses reproducible shuffled K-fold cross-validation to compare candidates.
- Trains four complementary candidates: a PyTorch MLP, random forest, Extra Trees, and histogram gradient boosting.
- Selects the model with the lowest mean validation MSE, reports MSE/R² variation across folds, then refits the winner on all supplied data.
- Records the number of samples used for the final refit as `training_samples`.
- Performs forward prediction: `X -> Y`.
- Performs bounded inverse solving: find distinct `X` values whose prediction is close to a target `Y`, with optimization diagnostics.
- Supports single- and multi-output targets.

## Quick start

Requires Python 3.10 or later.

```powershell
git clone https://github.com/kfat77/universal-adaptive-black-box.git
cd universal-adaptive-black-box
python -m pip install -r requirements.txt
python main.py
```

`main.py` generates noisy sine-wave observations, trains the engine, predicts at a new input, and finds two inputs that produce a target output close to `0.5`.

## Use with your data

Your arrays must contain only finite numeric values. Rows are observations; columns are features or output dimensions.

```python
from pathlib import Path

import numpy as np

from src.core_engine import AdaptiveBlackBox
from src.data_loader import load_tabular_data
from src.forward_solver import ForwardSolver
from src.inverse_solver import InverseSolver

# Example: 1,000 observations, 3 numerical inputs, and 2 outputs.
# CSV or Excel: select targets by column name. All other columns become features.
dataset = load_tabular_data("experiment.xlsx", target_columns=["yield", "purity"])
X, Y = dataset.X, dataset.Y
print(dataset.feature_names, dataset.target_names)
artifact_path = Path("artifacts/my_model.joblib")

# Train and persist the validation-selected model.
engine = AdaptiveBlackBox(epochs=500).fit(X, Y)
print(engine.model_name)
print(engine.metrics)
engine.save(artifact_path)

# Forward solve: predict outputs from new inputs.
forward = ForwardSolver(str(artifact_path))
Y_prediction = forward.predict(np.array([[1.2, -0.4, 5.0]]))

# Inverse solve: search within meaningful input bounds for a desired output.
inverse = InverseSolver(str(artifact_path))
solutions = inverse.inverse_solve(
    Y_target=np.array([10.0, 0.75]),
    x_bounds=[(0.0, 3.0), (-1.0, 1.0), (0.0, 10.0)],
    n_solutions=3,
    min_separation=0.05,
)
for solution in solutions:
    print(solution)
```

`load_tabular_data` supports `.csv`, `.xls`, and `.xlsx`. Use `feature_columns=[...]` to select and order input columns explicitly, or omit it to use every non-target column. Selected columns must be numerical and finite.

## How inverse solving works

Most learned black-box models cannot be algebraically inverted. For a requested target `Y_target`, the engine minimizes:

```text
mean((predict(X) - Y_target)²)
```

within the `x_bounds` supplied for every input dimension. It uses differential evolution to search the global bounded space, then L-BFGS-B to refine each candidate locally. Repeated searches may find different valid inputs, which is expected when the forward relationship is not one-to-one.

Choose bounds that reflect physically or operationally feasible inputs. The solver cannot distinguish an implausible solution unless that constraint is expressed through the bounds.

Each returned solution contains `x`, `predicted_y`, residual `mse`, a `success` flag, total objective `evaluations`, the search `attempt`, and the optimizer `message`. Inspect `success` and `mse` before using a solution. `min_separation` prevents near-duplicate inputs using Euclidean distance in the original input units; fewer than `n_solutions` may be returned when the bounded domain does not contain enough distinct candidates.

## API reference

| Component | Method | Purpose |
| --- | --- | --- |
| `AdaptiveBlackBox` | `fit(X, Y, validation_folds=3)` | Cross-validate all candidates and select by mean validation MSE. |
| `AdaptiveBlackBox` | `predict(X_new)` | Predict output rows from input rows. |
| `AdaptiveBlackBox` | `save(path)` / `load(path)` | Persist and restore the selected model and scalers. |
| `ForwardSolver` | `predict(X_new)` | Load an artifact and perform forward prediction. |
| `InverseSolver` | `inverse_solve(Y_target, x_bounds, n_solutions=1, min_separation=...)` | Return distinct candidate inputs, predictions, residual MSE, and diagnostics. |

## Project layout

```text
├── main.py                 # End-to-end sine-data demonstration
├── requirements.txt        # Runtime dependencies
├── LICENSE
├── .github/workflows/test.yml
├── src/
│   ├── core_engine.py      # Training, validation, selection, persistence
│   ├── data_loader.py      # CSV and Excel loading with numerical validation
│   ├── forward_solver.py   # Forward prediction wrapper
│   └── inverse_solver.py   # Bounded inverse optimization
└── tests/
    ├── test_data_loader.py
    ├── test_inverse_solver.py
    └── test_single_output.py
```

## Important limitations

- The selected model is chosen only from the four included numerical regression candidates; optional symbolic regression and external gradient-boosting libraries are intentionally out of scope for this lightweight release.
- Cross-validation estimates quality only when its held-out folds resemble future data; it does not replace an independent test set for high-stakes work.
- An inverse solution is an input that fits the learned model, not proof that it is physically unique or feasible.
- Saved artifacts use Python pickle. Load only artifacts that you created or trust.

## Verification

Run the included regression check:

```powershell
python -m unittest discover -s tests
```

## License

This project is released under the [MIT License](LICENSE).
