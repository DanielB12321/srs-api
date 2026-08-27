"""Offline training for the XGBoost and SVM similarity ensemble."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from ..algorithms.ml_ensemble import EXPECTED_FEATURE_COUNT, make_pair_features
from ..preprocessing import describe, prepare_vectors, resolve_options


RANDOM_STATE = 42
FINAL_TEST_RANDOM_STATE = 2026
CV_RANDOM_STATE = 2027
N_CV_SPLITS = 5
PAIRS_PER_SAMPLE = 12

TARGET_COLUMN = "Class"
GROUP_COLUMN = "Three Character Code"
DEPOSIT_NAME_COLUMN = "Deposit Name"

MIN_CLASS_SAMPLES = 20
MIN_CLASS_DEPOSITS = 5

NON_FEATURE_COLUMNS = {
    "Sample",
    "SAMPLE CODE",
    "Deposit Name",
    "Donor",
    "Three Character Code",
    "Class",
    "Sub Class",
    "Simplified Class",
    "NUM",
}

SVM_CANDIDATES = [
    {"svm_name": "rbf_c1_scale", "C": 1.0, "gamma": "scale"},
    {"svm_name": "rbf_c3_scale", "C": 3.0, "gamma": "scale"},
    {"svm_name": "rbf_c10_scale", "C": 10.0, "gamma": "scale"},
    {"svm_name": "rbf_c3_gamma_0_01", "C": 3.0, "gamma": 0.01},
]

XGB_WEIGHTS = [0.0, 0.20, 0.40, 0.50, 0.60, 0.80, 1.0]


def load_data(path: Path, sheet_name: str | None = None) -> pd.DataFrame:
    """Load a CSV or Excel training file and clean its class labels."""
    if not path.exists():
        raise FileNotFoundError(f"Training dataset not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        dataframe = pd.read_csv(path)
    elif suffix in {".xlsx", ".xls"}:
        dataframe = pd.read_excel(path, sheet_name=sheet_name or 0)
    else:
        raise ValueError("Training input must be CSV or Excel.")

    dataframe.columns = dataframe.columns.astype(str).str.strip()
    return clean_labels(dataframe)


def clean_labels(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Remove incomplete labels and classes with too little training data."""
    required = {TARGET_COLUMN, GROUP_COLUMN, DEPOSIT_NAME_COLUMN}
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    cleaned = dataframe.copy()
    for column in required:
        cleaned[column] = cleaned[column].astype("string").str.strip()

    cleaned = cleaned.dropna(subset=[TARGET_COLUMN, GROUP_COLUMN])
    cleaned = cleaned[
        (cleaned[TARGET_COLUMN] != "")
        & (cleaned[GROUP_COLUMN] != "")
    ]

    class_summary = cleaned.groupby(TARGET_COLUMN).agg(
        samples=(TARGET_COLUMN, "size"),
        deposits=(GROUP_COLUMN, "nunique"),
    )
    eligible = class_summary[
        (class_summary["samples"] >= MIN_CLASS_SAMPLES)
        & (class_summary["deposits"] >= MIN_CLASS_DEPOSITS)
    ].index

    cleaned = cleaned[cleaned[TARGET_COLUMN].isin(eligible)].reset_index(drop=True)
    if cleaned.empty:
        raise ValueError("No classes satisfy the minimum sample/deposit thresholds.")

    return cleaned


def identify_elements(dataframe: pd.DataFrame) -> list[str]:
    """Return columns containing usable numeric element values."""
    elements: list[str] = []
    for column in dataframe.columns:
        if column in NON_FEATURE_COLUMNS:
            continue
        numeric = pd.to_numeric(dataframe[column], errors="coerce")
        if numeric.notna().any():
            dataframe[column] = numeric
            elements.append(column)

    if not elements:
        raise ValueError("No numeric geochemical element columns were found.")
    return elements


def row_values(row: pd.Series, elements: list[str]) -> dict[str, float]:
    """Extract the available element values from one training row."""
    values: dict[str, float] = {}
    for element in elements:
        value = row[element]
        if pd.notna(value):
            values[element] = float(value)
    return values


def make_pair_examples(
    dataframe: pd.DataFrame,
    indices: np.ndarray,
    rng: np.random.Generator,
) -> list[tuple[int, int, int]]:
    """Create balanced positive/negative pairs within one deposit split.

    Positive pairs have the same ``Class`` but are deliberately chosen from
    different deposits. Negative pairs have different classes. This asks the
    ensemble to learn class-level geochemical similarity rather than memorising
    an individual deposit.
    """
    indices = np.asarray(indices, dtype=int)
    by_class: dict[str, list[int]] = {}

    for index in indices:
        label = str(dataframe.at[index, TARGET_COLUMN])
        by_class.setdefault(label, []).append(int(index))

    labels = sorted(by_class)
    half = max(1, PAIRS_PER_SAMPLE // 2)
    pairs: list[tuple[int, int, int]] = []

    for anchor in indices:
        anchor = int(anchor)
        label = str(dataframe.at[anchor, TARGET_COLUMN])
        deposit = str(dataframe.at[anchor, GROUP_COLUMN])

        positives = [
            candidate
            for candidate in by_class[label]
            if candidate != anchor
            and str(dataframe.at[candidate, GROUP_COLUMN]) != deposit
        ]
        negatives = [
            candidate
            for other_label in labels
            if other_label != label
            for candidate in by_class[other_label]
        ]

        if positives:
            chosen = rng.choice(
                positives,
                size=half,
                replace=len(positives) < half,
            )
            pairs.extend((anchor, int(candidate), 1) for candidate in chosen)

        if negatives:
            chosen = rng.choice(
                negatives,
                size=half,
                replace=len(negatives) < half,
            )
            pairs.extend((anchor, int(candidate), 0) for candidate in chosen)

    if not pairs:
        raise ValueError("Could not create positive/negative training pairs.")

    rng.shuffle(pairs)
    return pairs


def pair_matrix(
    dataframe: pd.DataFrame,
    pairs: list[tuple[int, int, int]],
    elements: list[str],
    options: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Build model features and targets for a list of sample pairs."""
    rows: list[np.ndarray] = []
    targets: list[int] = []

    for left_index, right_index, target in pairs:
        left_values = row_values(dataframe.iloc[left_index], elements)
        right_values = row_values(dataframe.iloc[right_index], elements)
        common_elements = set(left_values) & set(right_values)
        if not common_elements:
            continue

        prepared = prepare_vectors(
            left_values,
            right_values,
            common_elements,
            options,
            set(),
        )
        if not prepared.input_vector:
            continue

        rows.append(make_pair_features(prepared))
        targets.append(target)

    if not rows:
        raise ValueError("Shared preprocessing produced no usable pair features.")

    X = np.vstack(rows)
    y = np.asarray(targets, dtype=int)

    if X.shape[1] != EXPECTED_FEATURE_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_FEATURE_COUNT} pair features, got {X.shape[1]}."
        )
    if set(np.unique(y)) != {0, 1}:
        raise RuntimeError("Pair dataset must contain both positive and negative labels.")

    return X, y


def make_final_split(
    dataframe: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a final test split with no deposit shared with development data."""
    labels = dataframe[TARGET_COLUMN].astype(str).to_numpy()
    groups = dataframe[GROUP_COLUMN].astype(str).to_numpy()

    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=FINAL_TEST_RANDOM_STATE,
    )
    development_indices, final_test_indices = next(
        splitter.split(dataframe, labels, groups=groups)
    )

    overlap = set(groups[development_indices]) & set(groups[final_test_indices])
    if overlap:
        raise RuntimeError(f"Deposit leakage in final split: {sorted(overlap)}")

    return development_indices, final_test_indices


def create_xgb_model() -> XGBClassifier:
    """Create the XGBoost model used for each fold and the final fit."""
    return XGBClassifier(
        objective="binary:logistic",
        n_estimators=900,
        learning_rate=0.04,
        max_depth=4,
        min_child_weight=2,
        subsample=0.85,
        colsample_bytree=0.80,
        reg_alpha=0.10,
        reg_lambda=6.0,
        gamma=0.05,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def create_svm_pipeline(parameters: dict) -> Pipeline:
    """Create the scaled RBF-SVM pipeline for one parameter set."""
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    C=parameters["C"],
                    gamma=parameters["gamma"],
                    probability=True,
                    cache_size=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def calculate_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """Calculate binary classification metrics at a 0.5 threshold."""
    predictions = (probabilities >= 0.5).astype(int)
    result = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro")),
    }
    if len(np.unique(y_true)) == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, probabilities))
    else:
        result["roc_auc"] = 0.0
    return result


def run_grouped_cross_validation(
    dataframe: pd.DataFrame,
    development_indices: np.ndarray,
    elements: list[str],
    options: dict,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Select SVM settings and ensemble weights using deposit-grouped folds."""
    development = dataframe.iloc[development_indices]
    development_labels = development[TARGET_COLUMN].astype(str).to_numpy()
    development_groups = development[GROUP_COLUMN].astype(str).to_numpy()

    splitter = StratifiedGroupKFold(
        n_splits=N_CV_SPLITS,
        shuffle=True,
        random_state=CV_RANDOM_STATE,
    )

    result_rows: list[dict] = []

    for fold_number, (train_pos, validation_pos) in enumerate(
        splitter.split(development, development_labels, groups=development_groups),
        start=1,
    ):
        train_indices = development_indices[train_pos]
        validation_indices = development_indices[validation_pos]

        train_deposits = set(dataframe.iloc[train_indices][GROUP_COLUMN].astype(str))
        validation_deposits = set(
            dataframe.iloc[validation_indices][GROUP_COLUMN].astype(str)
        )
        overlap = train_deposits & validation_deposits
        if overlap:
            raise RuntimeError(
                f"Deposit leakage in CV fold {fold_number}: {sorted(overlap)}"
            )

        train_rng = np.random.default_rng(RANDOM_STATE + fold_number)
        validation_rng = np.random.default_rng(RANDOM_STATE + 100 + fold_number)

        train_pairs = make_pair_examples(dataframe, train_indices, train_rng)
        validation_pairs = make_pair_examples(
            dataframe, validation_indices, validation_rng
        )

        X_train, y_train = pair_matrix(dataframe, train_pairs, elements, options)
        X_validation, y_validation = pair_matrix(
            dataframe, validation_pairs, elements, options
        )

        xgb_model = create_xgb_model()
        xgb_model.fit(X_train, y_train)
        xgb_probabilities = xgb_model.predict_proba(X_validation)[:, 1]

        for svm_parameters in SVM_CANDIDATES:
            svm_pipeline = create_svm_pipeline(svm_parameters)
            svm_pipeline.fit(X_train, y_train)
            svm_probabilities = svm_pipeline.predict_proba(X_validation)[:, 1]

            for xgb_weight in XGB_WEIGHTS:
                svm_weight = 1.0 - xgb_weight
                combined = (
                    xgb_weight * xgb_probabilities
                    + svm_weight * svm_probabilities
                )
                result_rows.append(
                    {
                        "fold": fold_number,
                        "svm_name": svm_parameters["svm_name"],
                        "svm_C": float(svm_parameters["C"]),
                        "svm_gamma": str(svm_parameters["gamma"]),
                        "xgb_weight": float(xgb_weight),
                        "svm_weight": float(svm_weight),
                        **calculate_metrics(y_validation, combined),
                        "training_pairs": int(len(y_train)),
                        "validation_pairs": int(len(y_validation)),
                        "training_deposits": int(len(train_deposits)),
                        "validation_deposits": int(len(validation_deposits)),
                    }
                )

    fold_results = pd.DataFrame(result_rows)
    summary = (
        fold_results.groupby(
            ["svm_name", "svm_C", "svm_gamma", "xgb_weight", "svm_weight"],
            as_index=False,
        )
        .agg(
            mean_macro_f1=("macro_f1", "mean"),
            std_macro_f1=("macro_f1", "std"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            std_balanced_accuracy=("balanced_accuracy", "std"),
            mean_accuracy=("accuracy", "mean"),
            mean_roc_auc=("roc_auc", "mean"),
        )
        .sort_values(
            ["mean_macro_f1", "mean_balanced_accuracy", "mean_roc_auc"],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    best = summary.iloc[0]
    selected = {
        "svm_name": str(best["svm_name"]),
        "svm_C": float(best["svm_C"]),
        "svm_gamma": (
            str(best["svm_gamma"])
            if str(best["svm_gamma"]) == "scale"
            else float(best["svm_gamma"])
        ),
        "xgb_weight": float(best["xgb_weight"]),
        "svm_weight": float(best["svm_weight"]),
    }
    return selected, fold_results, summary


def fit_final_models(
    dataframe: pd.DataFrame,
    development_indices: np.ndarray,
    final_test_indices: np.ndarray,
    elements: list[str],
    options: dict,
    selected: dict,
):
    """Fit both models on development data and evaluate the held-out split."""
    development_rng = np.random.default_rng(RANDOM_STATE + 500)
    final_rng = np.random.default_rng(RANDOM_STATE + 600)

    development_pairs = make_pair_examples(
        dataframe, development_indices, development_rng
    )
    final_test_pairs = make_pair_examples(dataframe, final_test_indices, final_rng)

    X_development, y_development = pair_matrix(
        dataframe, development_pairs, elements, options
    )
    X_final, y_final = pair_matrix(dataframe, final_test_pairs, elements, options)

    xgb_model = create_xgb_model()
    svm_pipeline = create_svm_pipeline(
        {"C": selected["svm_C"], "gamma": selected["svm_gamma"]}
    )

    xgb_model.fit(X_development, y_development)
    svm_pipeline.fit(X_development, y_development)

    xgb_probabilities = xgb_model.predict_proba(X_final)[:, 1]
    svm_probabilities = svm_pipeline.predict_proba(X_final)[:, 1]
    combined = (
        selected["xgb_weight"] * xgb_probabilities
        + selected["svm_weight"] * svm_probabilities
    )

    return (
        xgb_model,
        svm_pipeline,
        calculate_metrics(y_final, combined),
        int(len(y_development)),
        int(len(y_final)),
    )


def train_from_file(
    *,
    input_path: Path,
    output_directory: Path,
    sheet_name: str | None = None,
    preprocessing_request: dict | None = None,
) -> dict:
    """Train the ensemble and write its models, metrics and manifest."""
    dataframe = load_data(input_path, sheet_name)
    elements = identify_elements(dataframe)

    # This is the same preprocessing resolver used by production algorithms.
    options = resolve_options(preprocessing_request)

    development_indices, final_test_indices = make_final_split(dataframe)

    selected, fold_results, cv_summary = run_grouped_cross_validation(
        dataframe,
        development_indices,
        elements,
        options,
    )

    (
        xgb_model,
        svm_pipeline,
        final_metrics,
        development_pair_count,
        final_pair_count,
    ) = fit_final_models(
        dataframe,
        development_indices,
        final_test_indices,
        elements,
        options,
        selected,
    )

    output_directory.mkdir(parents=True, exist_ok=True)

    xgb_model.save_model(output_directory / "xgb_model.json")
    joblib.dump(svm_pipeline, output_directory / "svm_pipeline.joblib")
    fold_results.to_csv(output_directory / "cv_fold_results.csv", index=False)
    cv_summary.to_csv(output_directory / "cv_summary.csv", index=False)

    development_deposits = set(
        dataframe.iloc[development_indices][GROUP_COLUMN].astype(str)
    )
    final_test_deposits = set(
        dataframe.iloc[final_test_indices][GROUP_COLUMN].astype(str)
    )

    manifest = {
        "algorithm_id": "xgboost_rbf_svm_ensemble",
        "version": "1.0.0",
        "task": "pairwise_same_deposit_class_similarity",
        "pair_target": "1=same Class across reference deposits; 0=different Class",
        "feature_count": EXPECTED_FEATURE_COUNT,
        "training_samples": int(len(development_indices)),
        "final_test_samples": int(len(final_test_indices)),
        "training_pairs": development_pair_count,
        "final_test_pairs": final_pair_count,
        "training_deposits": int(len(development_deposits)),
        "final_test_deposits": int(len(final_test_deposits)),
        "classes": sorted(dataframe[TARGET_COLUMN].astype(str).unique().tolist()),
        "elements_seen_during_training": elements,
        "selected_ensemble": selected,
        "selection_metric": "mean_macro_f1",
        "cross_validation": cv_summary.iloc[0].to_dict(),
        "final_test_metrics": final_metrics,
        "preprocessing": describe(options, elements),
        "notes": (
            "Geochemical preprocessing is provided exclusively by the shared "
            "SRS preprocessing head. XGBoost and RBF-SVM receive the same 20 "
            "pair-summary features derived from PreparedVectors. The SVM's "
            "StandardScaler is estimator-internal numerical scaling."
        ),
    }

    with (output_directory / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    return manifest
