"""Inference adapter for the trained XGBoost and SVM ensemble."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from .base import PairwiseSimilarity


MODEL_DIRECTORY = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "ml_ensemble"
)

EXPECTED_FEATURE_COUNT = 20


class AlgorithmUnavailableError(RuntimeError):
    """Raised when the optional ML model cannot be used on this installation."""


def _load_numpy():
    """Import NumPy only when the ML algorithm is used."""
    try:
        import numpy as np
    except (ImportError, OSError) as exc:
        raise AlgorithmUnavailableError(
            "The ML ensemble is unavailable because its optional dependencies "
            "are not installed. Install the packages from requirements.txt."
        ) from exc
    return np


def make_pair_features(prepared):
    """Convert one prepared sample pair to the model's 20 input features."""
    np = _load_numpy()
    left = np.asarray(prepared.input_vector, dtype=float)
    right = np.asarray(prepared.reference_vector, dtype=float)

    if left.size == 0 or right.size == 0:
        return np.zeros(EXPECTED_FEATURE_COUNT, dtype=float)

    if left.shape != right.shape:
        raise ValueError("Prepared input/reference vectors must have equal length.")

    differences = left - right
    absolute = np.abs(differences)
    squared = differences ** 2
    midpoint = (left + right) / 2.0

    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    cosine = (
        float(np.dot(left, right) / (left_norm * right_norm))
        if left_norm > 0.0 and right_norm > 0.0
        else 0.0
    )

    correlation = 0.0
    if left.size >= 2 and np.std(left) > 0.0 and np.std(right) > 0.0:
        correlation = float(np.corrcoef(left, right)[0, 1])
        if not np.isfinite(correlation):
            correlation = 0.0

    weights = getattr(prepared, "weights", None)
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        weight_sum = float(weights.sum())
        if weight_sum > 0.0:
            weighted_abs = float(np.sum(weights * absolute) / weight_sum)
            weighted_sq = float(np.sum(weights * squared) / weight_sum)
        else:
            weighted_abs = float(np.mean(absolute))
            weighted_sq = float(np.mean(squared))
    else:
        weighted_abs = float(np.mean(absolute))
        weighted_sq = float(np.mean(squared))

    imputed = list(getattr(prepared, "imputed", None) or [])
    imputed_fraction = (
        float(sum(bool(value) for value in imputed) / len(imputed))
        if imputed
        else 0.0
    )

    features = np.asarray(
        [
            float(left.size),
            float(np.mean(absolute)),
            float(np.median(absolute)),
            float(np.sqrt(np.mean(squared))),
            float(np.max(absolute)),
            float(np.quantile(absolute, 0.25)),
            float(np.quantile(absolute, 0.75)),
            float(np.mean(differences)),
            float(np.std(differences)),
            weighted_abs,
            float(np.sqrt(max(weighted_sq, 0.0))),
            cosine,
            correlation,
            float(np.mean(left)),
            float(np.std(left)),
            float(np.mean(right)),
            float(np.std(right)),
            float(np.mean(midpoint)),
            float(np.std(midpoint)),
            imputed_fraction,
        ],
        dtype=float,
    )

    if features.size != EXPECTED_FEATURE_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_FEATURE_COUNT} ML features, got {features.size}."
        )

    return features


@lru_cache(maxsize=1)
def _load_artifacts():
    """Load trained model files once per Python worker process."""
    try:
        import joblib
        from xgboost import XGBClassifier
    except (ImportError, OSError) as exc:
        raise AlgorithmUnavailableError(
            "The ML ensemble is unavailable because its optional dependencies "
            "are not installed. Install the packages from requirements.txt."
        ) from exc

    paths = {
        "xgb": MODEL_DIRECTORY / "xgb_model.json",
        "svm": MODEL_DIRECTORY / "svm_pipeline.joblib",
        "manifest": MODEL_DIRECTORY / "manifest.json",
    }

    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise AlgorithmUnavailableError(
            "The ML ensemble is unavailable because model files are missing: "
            + ", ".join(missing)
        )

    try:
        xgb_model = XGBClassifier()
        xgb_model.load_model(paths["xgb"])
        svm_pipeline = joblib.load(paths["svm"])

        with paths["manifest"].open("r", encoding="utf-8") as file:
            manifest = json.load(file)
    except Exception as exc:
        raise AlgorithmUnavailableError(
            f"The ML ensemble model files could not be loaded: {exc}"
        ) from exc

    feature_count = int(manifest.get("feature_count", -1))
    if feature_count != EXPECTED_FEATURE_COUNT:
        raise AlgorithmUnavailableError(
            "The ML ensemble feature definition does not match the current code: "
            f"artifact={feature_count}, runtime={EXPECTED_FEATURE_COUNT}."
        )

    return xgb_model, svm_pipeline, manifest


class XgbSvmEnsembleSimilarity(PairwiseSimilarity):
    """Estimate the probability that two samples share a deposit class."""

    id = "xgboost_rbf_svm_ensemble"
    version = "1.0.0"
    capabilities = frozenset()

    def score_vectors(self, prepared):
        np = _load_numpy()
        xgb_model, svm_pipeline, manifest = _load_artifacts()
        features = make_pair_features(prepared).reshape(1, -1)

        xgb_probability = float(xgb_model.predict_proba(features)[0, 1])
        svm_probability = float(svm_pipeline.predict_proba(features)[0, 1])

        selected = manifest["selected_ensemble"]
        xgb_weight = float(selected["xgb_weight"])
        svm_weight = float(selected["svm_weight"])

        similarity = (
            xgb_weight * xgb_probability
            + svm_weight * svm_probability
        )

        return float(np.clip(similarity, 0.0, 1.0))
