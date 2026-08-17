"""Focused tests for the ML algorithm adapter.

These tests belong in the existing SRS test suite, where ``srs.preprocessing``
and the algorithm base classes are already available.
"""

from types import SimpleNamespace

import numpy as np

from srs.algorithms import ml_ensemble
from srs.algorithms.ml_ensemble import (
    EXPECTED_FEATURE_COUNT,
    XgbSvmEnsembleSimilarity,
    make_pair_features,
)


def prepared_example():
    return SimpleNamespace(
        input_vector=[1.0, 2.0, 3.0, 4.0],
        reference_vector=[1.1, 1.9, 2.8, 4.2],
        weights=None,
        imputed=[False, False, True, False],
    )


def test_pair_features_are_fixed_length_and_finite():
    features = make_pair_features(prepared_example())
    assert features.shape == (EXPECTED_FEATURE_COUNT,)
    assert np.isfinite(features).all()


def test_ensemble_returns_common_similarity_range(monkeypatch):
    class FakeModel:
        def __init__(self, positive_probability):
            self.positive_probability = positive_probability

        def predict_proba(self, X):
            p = self.positive_probability
            return np.asarray([[1.0 - p, p] for _ in range(len(X))])

    manifest = {
        "selected_ensemble": {
            "xgb_weight": 0.6,
            "svm_weight": 0.4,
        }
    }

    monkeypatch.setattr(
        ml_ensemble,
        "_load_artifacts",
        lambda: (FakeModel(0.8), FakeModel(0.5), manifest),
    )

    score = XgbSvmEnsembleSimilarity().score_vectors(prepared_example())

    assert score == 0.68
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
