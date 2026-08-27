"""Tests for the optional ML algorithm adapter."""

import builtins
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest import skipUnless
from unittest.mock import patch

from django.test import SimpleTestCase

from srs.algorithms import get_algorithm, ml_ensemble
from srs.algorithms.ml_ensemble import (
    AlgorithmUnavailableError,
    EXPECTED_FEATURE_COUNT,
    XgbSvmEnsembleSimilarity,
    make_pair_features,
)


HAS_NUMPY = importlib.util.find_spec("numpy") is not None


def prepared_example():
    return SimpleNamespace(
        input_vector=[1.0, 2.0, 3.0, 4.0],
        reference_vector=[1.1, 1.9, 2.8, 4.2],
        weights=None,
        imputed=[False, False, True, False],
    )


class MlEnsembleAdapterTests(SimpleTestCase):
    def test_registry_loads_without_importing_optional_packages(self):
        algorithm = get_algorithm("xgboost_rbf_svm_ensemble")

        self.assertIsInstance(algorithm, XgbSvmEnsembleSimilarity)

    def test_missing_numpy_has_a_clear_unavailable_error(self):
        real_import = builtins.__import__

        def import_without_numpy(name, *args, **kwargs):
            if name == "numpy":
                raise ModuleNotFoundError("No module named 'numpy'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=import_without_numpy):
            with self.assertRaisesRegex(
                AlgorithmUnavailableError,
                "optional dependencies",
            ):
                make_pair_features(prepared_example())

    def test_missing_model_files_have_a_clear_unavailable_error(self):
        fake_joblib = ModuleType("joblib")
        fake_xgboost = ModuleType("xgboost")
        fake_xgboost.XGBClassifier = object
        missing_directory = Path("model-directory-that-does-not-exist")

        ml_ensemble._load_artifacts.cache_clear()
        with patch.dict(
            sys.modules,
            {"joblib": fake_joblib, "xgboost": fake_xgboost},
        ), patch.object(ml_ensemble, "MODEL_DIRECTORY", missing_directory):
            with self.assertRaisesRegex(
                AlgorithmUnavailableError,
                "model files are missing",
            ):
                ml_ensemble._load_artifacts()
        ml_ensemble._load_artifacts.cache_clear()

    @skipUnless(HAS_NUMPY, "NumPy is not installed in this environment.")
    def test_pair_features_are_fixed_length_and_finite(self):
        import numpy as np

        features = make_pair_features(prepared_example())

        self.assertEqual(features.shape, (EXPECTED_FEATURE_COUNT,))
        self.assertTrue(np.isfinite(features).all())

    @skipUnless(HAS_NUMPY, "NumPy is not installed in this environment.")
    def test_ensemble_returns_common_similarity_range(self):
        import numpy as np

        class FakeModel:
            def __init__(self, positive_probability):
                self.positive_probability = positive_probability

            def predict_proba(self, rows):
                probability = self.positive_probability
                return np.asarray([
                    [1.0 - probability, probability]
                    for _ in range(len(rows))
                ])

        manifest = {
            "selected_ensemble": {
                "xgb_weight": 0.6,
                "svm_weight": 0.4,
            }
        }

        with patch.object(
            ml_ensemble,
            "_load_artifacts",
            return_value=(FakeModel(0.8), FakeModel(0.5), manifest),
        ):
            score = XgbSvmEnsembleSimilarity().score_vectors(prepared_example())

        self.assertAlmostEqual(score, 0.68)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
