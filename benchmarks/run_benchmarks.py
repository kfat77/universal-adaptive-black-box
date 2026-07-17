"""Run deterministic local benchmark datasets and write actual CSV results."""

import csv
from pathlib import Path

from datasets import benchmark_datasets

from adaptive_surrogate import AdaptiveBlackBox


def main() -> None:
    rows: list[dict[str, str | float | bool]] = []
    for dataset_name, (x, y) in benchmark_datasets().items():
        engine = AdaptiveBlackBox(epochs=40).fit(x, y, validation_folds=2)
        for candidate, metrics in engine.metrics.items():
            rows.append(
                {
                    "dataset": dataset_name,
                    "candidate": candidate,
                    "selected": bool(metrics.get("selected", False)),
                    "baseline": bool(metrics["baseline"]),
                    "rmse": float(metrics["rmse"]),
                    "nrmse": float(metrics["nrmse"]),
                    "mae": float(metrics["mae"]),
                    "r2": float(metrics["r2"]),
                    "training_seconds": float(metrics["training_seconds"]),
                    "inference_seconds": float(metrics["inference_seconds"]),
                }
            )
    output = Path(__file__).parent / "results" / "example_results.csv"
    output.parent.mkdir(exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} measured candidate rows to {output}")


if __name__ == "__main__":
    main()
