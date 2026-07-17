import unittest

import numpy as np

from adaptive_surrogate import AdaptiveBlackBox


class SearchModeTest(unittest.TestCase):
    def test_balanced_search_records_reproducible_inner_search_details(self) -> None:
        x = np.linspace(-1.0, 1.0, 24).reshape(-1, 1)
        engine = AdaptiveBlackBox(epochs=2).fit(
            x, x**2, validation_folds=2, search_mode="balanced"
        )
        self.assertEqual(engine.search_mode, "balanced")
        self.assertIn("ridge", engine.search_details)
        self.assertEqual(engine.search_details["ridge"]["budget"], 3)

    def test_unknown_search_mode_is_rejected(self) -> None:
        x = np.linspace(-1.0, 1.0, 12).reshape(-1, 1)
        with self.assertRaisesRegex(ValueError, "search_mode"):
            AdaptiveBlackBox(epochs=1).fit(x, x, search_mode="slow")
