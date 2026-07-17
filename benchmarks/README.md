# Benchmarks

`run_benchmarks.py` evaluates every candidate, including baselines, on local Friedman1, Diabetes, synthetic linear/nonlinear, and multi-output datasets. It writes measured results to `results/example_results.csv`. It is a reproducible smoke benchmark, not evidence of performance on a domain dataset.

```powershell
python benchmarks/run_benchmarks.py
```

For a real comparison, use a held-out dataset with domain-appropriate splits, record package versions and random seeds, and report baseline as well as selected-model performance.
