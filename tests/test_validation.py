"""Validation strategy behavior tests."""

import unittest

import numpy as np

from src.validation import build_splits


class ValidationTest(unittest.TestCase):
    def test_group_kfold_never_splits_a_group(self) -> None:
        groups = np.repeat(np.arange(4), 3)
        splits = list(build_splits("group_kfold", len(groups), 2, 7, groups=groups))
        for train, validation in splits:
            self.assertFalse(set(groups[train]) & set(groups[validation]))

    def test_time_series_preserves_temporal_order(self) -> None:
        for train, validation in build_splits("time_series", 12, 3, 7):
            self.assertLess(max(train), min(validation))
