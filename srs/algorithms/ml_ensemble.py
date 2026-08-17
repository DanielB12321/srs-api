"""Learned pairwise similarity using a soft-voting XGBoost + RBF-SVM ensemble.

This module is intentionally inference-only. Raw geochemical preprocessing is
owned by the shared SRS preprocessing head in ``srs.preprocessing`` and is
applied by ``PairwiseSimilarity`` before ``score_vectors`` is called.

The ensemble is trained offline with the matching management command and saves
its learned parameters under ``srs/models/ml_ensemble``.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from xgboost import XGBClassifier

from .base import PairwiseSimilarity


MODEL_DIRECTORY = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "ml_ensemble"
)

EXPECTED_FEATURE_COUNT = 20


def make_pair_features(prepared) -> np.ndarray:
    """Convert one already-preprocessed pair to a fixed-length ML feature row.

    ``prepared`` is the object produced by the shared preprocessing head. These
    are model features derived from that output; this function does not perform
    raw-value imputation, censor handling, log transformation, CLR, selection,
    or weighting policy resolution.
    """
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
    paths = {
        "xgb": MODEL_DIRECTORY / "xgb_model.json",
        "svm": MODEL_DIRECTORY / "svm_pipeline.joblib",
        "manifest": MODEL_DIRECTORY / "manifest.json",
    }

    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "ML ensemble has not been trained/deployed. Missing: "
            + ", ".join(missing)
        )

    xgb_model = XGBClassifier()
    xgb_model.load_model(paths["xgb"])
    svm_pipeline = joblib.load(paths["svm"])

    with paths["manifest"].open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    feature_count = int(manifest.get("feature_count", -1))
    if feature_count != EXPECTED_FEATURE_COUNT:
        raise RuntimeError(
            "ML ensemble artifact feature definition does not match runtime "
            f"code: artifact={feature_count}, runtime={EXPECTED_FEATURE_COUNT}."
        )

    return xgb_model, svm_pipeline, manifest


class XgbSvmEnsembleSimilarity(PairwiseSimilarity):
    """Learn whether an input/reference pair represents the same deposit class.

    XGBoost and an RBF-SVM independently estimate the probability that the two
    shared-preprocessed signatures belong to the same labelled mineral-deposit
    class. Their probabilities are combined with the soft-voting weights chosen
    during deposit-grouped cross-validation.

    Normalisation: the returned similarity is the ensemble probability in
    ``[0, 1]``. A larger value means stronger learned evidence that the input
    and reference signatures correspond to the same deposit class.
    """

    id = "xgboost_rbf_svm_ensemble"
    version = "1.0.0"
    capabilities = frozenset()

    def score_vectors(self, prepared):
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
