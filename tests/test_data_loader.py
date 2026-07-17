"""Public-interface tests for tabular dataset loading."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data_loader import load_tabular_data


class DataLoaderTest(unittest.TestCase):
    def test_loads_csv_and_preserves_column_metadata(self) -> None:
        frame = pd.DataFrame({
            "temperature": [10.0, 12.0, 14.0],
            "pressure": [1.0, 1.2, 1.4],
            "yield": [0.3, 0.5, 0.8],
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.csv"
            frame.to_csv(path, index=False)
            dataset = load_tabular_data(path, target_columns="yield")

        self.assertEqual(dataset.feature_names, ("temperature", "pressure"))
        self.assertEqual(dataset.target_names, ("yield",))
        self.assertEqual(dataset.X.shape, (3, 2))
        self.assertEqual(dataset.Y.shape, (3, 1))

    def test_rejects_missing_or_non_numeric_columns(self) -> None:
        frame = pd.DataFrame({"x": [1.0, 2.0], "label": ["a", "b"]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.csv"
            frame.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_tabular_data(path, target_columns="missing")
            with self.assertRaises(ValueError):
                load_tabular_data(path, target_columns="label")

    def test_loads_excel_with_explicit_feature_columns(self) -> None:
        frame = pd.DataFrame({"x1": [1.0, 2.0], "x2": [3.0, 4.0], "y": [5.0, 6.0]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.xlsx"
            frame.to_excel(path, index=False)
            dataset = load_tabular_data(path, target_columns="y", feature_columns=["x2", "x1"])

        self.assertEqual(dataset.feature_names, ("x2", "x1"))
        self.assertEqual(dataset.X[0].tolist(), [3.0, 1.0])

    def test_rejects_non_finite_values(self) -> None:
        frame = pd.DataFrame({"x": [1.0, float("nan")], "y": [2.0, 3.0]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.csv"
            frame.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_tabular_data(path, target_columns="y")


if __name__ == "__main__":
    unittest.main()
