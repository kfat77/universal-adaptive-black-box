# Examples

Run examples from the repository root after installing the package:

```powershell
python -m pip install -e .[excel]
python examples/basic_workflow.py
python examples/linear_baseline.py
python examples/multi_output.py
python examples/constrained_inverse_design.py
python examples/uncertainty_ood_active_learning.py
python examples/pareto_design.py
```

Each script generates synthetic data locally. They illustrate API usage; they do not validate a model for a real experiment.
