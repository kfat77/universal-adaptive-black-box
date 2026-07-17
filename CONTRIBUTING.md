# Contributing

Contributions are welcome. Please open an issue before proposing a substantial API or algorithm change.

Before opening a pull request, run:

```powershell
python -m ruff check src tests
python -m ruff format --check src tests
python -m mypy src/adaptive_surrogate
python -m pytest
```

Keep changes focused, add tests for new behavior, and do not claim unsupported scientific guarantees.
