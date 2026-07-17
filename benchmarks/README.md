# Benchmarks

`run_benchmarks.py` compares the toolkit's selected surrogate against the included dummy baseline on small synthetic numerical problems. It is a reproducible smoke benchmark, not evidence of performance on a domain dataset.

```powershell
python benchmarks/run_benchmarks.py
```

For a real comparison, use a held-out dataset with domain-appropriate splits, record package versions and random seeds, and report baseline as well as selected-model performance.
