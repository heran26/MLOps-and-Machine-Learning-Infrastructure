"""
Basic sanity tests. Run with: pytest tests/
These don't require downloading the full dataset -- they just check the
preprocessing code is wired up correctly before a full training run.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocess import build_preprocessor, NUMERIC_FEATURES, CATEGORICAL_FEATURES


def test_preprocessor_builds():
    preprocessor = build_preprocessor()
    assert preprocessor is not None


def test_feature_lists_not_empty():
    assert len(NUMERIC_FEATURES) > 0
    assert len(CATEGORICAL_FEATURES) > 0
