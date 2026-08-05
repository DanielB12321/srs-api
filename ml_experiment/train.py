from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    top_k_accuracy_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
N_SPLITS = 5

ML_DIRECTORY = Path(__file__).resolve().parent
DATA_DIRECTORY = ML_DIRECTORY / "data"
ARTIFACT_DIRECTORY = ML_DIRECTORY / "artifacts"

DATA_FILE_NAME = "OSNACA-Data-1.xlsx"
SHEET_NAME = "Data 24 clip"

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


# ---------------------------------------------------------------------------
# Loading and cleaning
# ---------------------------------------------------------------------------

def load_osnaca() -> pd.DataFrame:
    data_path = DATA_DIRECTORY / DATA_FILE_NAME

    if not data_path.exists():
        raise FileNotFoundError(
            f"Could not find the OSNACA file at:\n{data_path}\n"
            "Update DATA_FILE_NAME at the top of train.py."
        )

    workbook = pd.ExcelFile(data_path)

    print("Workbook sheets:")
    for sheet in workbook.sheet_names:
        print(f"  - {sheet}")

    if SHEET_NAME not in workbook.sheet_names:
        raise ValueError(
            f"Sheet '{SHEET_NAME}' was not found. "
            f"Available sheets: {workbook.sheet_names}"
        )

    dataframe = pd.read_excel(data_path, sheet_name=SHEET_NAME)
    dataframe.columns = dataframe.columns.astype(str).str.strip()

    print(f"\nLoaded {len(dataframe):,} rows")
    print(f"Loaded {len(dataframe.columns):,} columns")

    return dataframe


def clean_labels(dataframe: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        TARGET_COLUMN,
        GROUP_COLUMN,
        DEPOSIT_NAME_COLUMN,
    }

    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    cleaned = dataframe.copy()

    for column in required_columns:
        cleaned[column] = (
            cleaned[column]
            .astype("string")
            .str.strip()
        )

    cleaned = cleaned.dropna(
        subset=[TARGET_COLUMN, GROUP_COLUMN]
    )

    cleaned = cleaned[
        (cleaned[TARGET_COLUMN] != "")
        & (cleaned[GROUP_COLUMN] != "")
    ]

    # Keep classes that have enough samples and independent deposits.
    class_summary = cleaned.groupby(TARGET_COLUMN).agg(
        samples=(TARGET_COLUMN, "size"),
        deposits=(GROUP_COLUMN, "nunique"),
    )

    eligible_classes = class_summary[
        (class_summary["samples"] >= MIN_CLASS_SAMPLES)
        & (class_summary["deposits"] >= MIN_CLASS_DEPOSITS)
    ].index

    cleaned = cleaned[
        cleaned[TARGET_COLUMN].isin(eligible_classes)
    ].reset_index(drop=True)

    print("\nEligible classes:")
    print(
        class_summary.loc[eligible_classes]
        .sort_values("samples", ascending=False)
    )

    print(f"\nRows retained: {len(cleaned):,}")
    print(
        f"Deposits retained: "
        f"{cleaned[GROUP_COLUMN].nunique():,}"
    )
    print(
        f"Classes retained: "
        f"{cleaned[TARGET_COLUMN].nunique():,}"
    )

    return cleaned


def identify_element_columns(
    dataframe: pd.DataFrame,
) -> list[str]:
    element_columns: list[str] = []

    for column in dataframe.columns:
        if column in NON_FEATURE_COLUMNS:
            continue

        numeric_values = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

        # Require at least some usable numeric measurements.
        if numeric_values.notna().sum() > 0:
            dataframe[column] = numeric_values
            element_columns.append(column)

    if not element_columns:
        raise ValueError("No numeric element columns were identified.")

    print(f"\nIdentified {len(element_columns)} element columns:")
    print(element_columns)

    return element_columns


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def prepare_raw_values(
    dataframe: pd.DataFrame,
    element_columns: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = dataframe[element_columns].to_numpy(
        dtype=float,
        copy=True,
    )

    censor_mask = np.isfinite(values) & (values < 0)
    missing_mask = ~np.isfinite(values)

    # Negative values represent below-detection measurements.
    values[censor_mask] = np.abs(values[censor_mask]) / 2.0

    # Logarithms require positive values.
    values[
        np.isfinite(values) & (values <= 0)
    ] = np.nan

    return values, censor_mask, missing_mask


def fit_training_medians(
    training_values: np.ndarray,
    element_columns: list[str],
) -> np.ndarray:
    medians = np.nanmedian(training_values, axis=0)

    invalid_indices = np.where(~np.isfinite(medians))[0]

    if len(invalid_indices) > 0:
        invalid_elements = [
            element_columns[index]
            for index in invalid_indices
        ]

        raise ValueError(
            "No valid training values were available for: "
            f"{invalid_elements}"
        )

    return medians


def apply_medians(
    values: np.ndarray,
    medians: np.ndarray,
) -> np.ndarray:
    result = values.copy()

    missing_rows, missing_columns = np.where(
        ~np.isfinite(result)
    )

    result[missing_rows, missing_columns] = (
        medians[missing_columns]
    )

    return np.clip(result, 1e-12, None)


def make_xgboost_features(
    values: np.ndarray,
    censor_mask: np.ndarray,
    missing_mask: np.ndarray,
) -> np.ndarray:
    log_values = np.log10(values)

    # The masks let the model distinguish measured, censored and missing data.
    return np.hstack([
        log_values,
        censor_mask.astype(np.float32),
        missing_mask.astype(np.float32),
    ])


# ---------------------------------------------------------------------------
# Weighting
# ---------------------------------------------------------------------------

def calculate_training_weights(
    labels: np.ndarray,
    deposit_ids: np.ndarray,
) -> np.ndarray:
    class_weights = compute_sample_weight(
        class_weight="balanced",
        y=labels,
    )

    deposit_counts = pd.Series(deposit_ids).value_counts()

    deposit_weights = np.asarray([
        1.0 / deposit_counts[deposit_id]
        for deposit_id in deposit_ids
    ])

    combined = class_weights * deposit_weights

    # Normalise so the average weight is one.
    return combined * len(combined) / combined.sum()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def create_model(number_of_classes: int) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=number_of_classes,
        n_estimators=700,
        learning_rate=0.035,
        max_depth=4,
        min_child_weight=2,
        subsample=0.85,
        colsample_bytree=0.80,
        reg_alpha=0.05,
        reg_lambda=4.0,
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def main() -> None:
    ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    dataframe = load_osnaca()
    dataframe = clean_labels(dataframe)
    element_columns = identify_element_columns(dataframe)

    labels = dataframe[TARGET_COLUMN].to_numpy()
    deposit_ids = dataframe[GROUP_COLUMN].to_numpy()

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # Use one complete fold as the first locked test set.
    train_indices, test_indices = next(
        splitter.split(
            dataframe,
            encoded_labels,
            groups=deposit_ids,
        )
    )

    train_data = dataframe.iloc[train_indices]
    test_data = dataframe.iloc[test_indices]

    y_train = encoded_labels[train_indices]
    y_test = encoded_labels[test_indices]

    train_deposits = deposit_ids[train_indices]
    test_deposits = deposit_ids[test_indices]

    overlap = set(train_deposits).intersection(test_deposits)

    if overlap:
        raise RuntimeError(
            f"Deposit leakage detected: {sorted(overlap)}"
        )

    print("\nSplit summary:")
    print(f"Training samples: {len(train_indices):,}")
    print(f"Test samples: {len(test_indices):,}")
    print(f"Training deposits: {len(set(train_deposits)):,}")
    print(f"Test deposits: {len(set(test_deposits)):,}")

    train_values, train_censored, train_missing = (
        prepare_raw_values(train_data, element_columns)
    )

    test_values, test_censored, test_missing = (
        prepare_raw_values(test_data, element_columns)
    )

    # Only training data are used to calculate medians.
    medians = fit_training_medians(
        train_values,
        element_columns,
    )

    train_values = apply_medians(train_values, medians)
    test_values = apply_medians(test_values, medians)

    X_train = make_xgboost_features(
        train_values,
        train_censored,
        train_missing,
    )

    X_test = make_xgboost_features(
        test_values,
        test_censored,
        test_missing,
    )

    training_weights = calculate_training_weights(
        y_train,
        train_deposits,
    )

    model = create_model(
        number_of_classes=len(label_encoder.classes_)
    )

    print("\nTraining XGBoost...")
    model.fit(
        X_train,
        y_train,
        sample_weight=training_weights,
    )

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    metrics = {
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, predictions)
        ),
        "macro_f1": float(
            f1_score(
                y_test,
                predictions,
                average="macro",
            )
        ),
        "top_3_accuracy": float(
            top_k_accuracy_score(
                y_test,
                probabilities,
                k=min(3, len(label_encoder.classes_)),
                labels=np.arange(
                    len(label_encoder.classes_)
                ),
            )
        ),
    }

    print("\nTest metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=np.arange(
                len(label_encoder.classes_)
            ),
            target_names=label_encoder.classes_,
            zero_division=0,
        )
    )

    print("\nNormalised confusion matrix:")

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=np.arange(len(label_encoder.classes_)),
        normalize="true",
    )

    confusion_dataframe = pd.DataFrame(
        matrix,
        index=label_encoder.classes_,
        columns=label_encoder.classes_,
    )

    print(confusion_dataframe.round(2).to_string())

    confusion_dataframe.to_csv(
        ARTIFACT_DIRECTORY / "confusion_matrix.csv"
    )

    report = classification_report(
        y_test,
        predictions,
        labels=np.arange(len(label_encoder.classes_)),
        target_names=label_encoder.classes_,
        zero_division=0,
        output_dict=True,
    )

    report_dataframe = pd.DataFrame(report).transpose()

    report_dataframe.to_csv(
        ARTIFACT_DIRECTORY / "classification_report.csv"
    )

    print("\nPer-class report:")
    print(report_dataframe.round(3).to_string())

    # Save the XGBoost model in its native JSON format.
    model.save_model(
        ARTIFACT_DIRECTORY / "xgb_model.json"
    )

    preprocessing_artifact = {
        "element_columns": element_columns,
        "medians": medians,
        "target_column": TARGET_COLUMN,
        "group_column": GROUP_COLUMN,
        "censored_policy": "absolute_detection_limit_divided_by_2",
        "feature_representation": "log10_values_with_censor_and_missing_masks",
    }

    joblib.dump(
        preprocessing_artifact,
        ARTIFACT_DIRECTORY / "preprocessing.joblib",
    )

    joblib.dump(
        label_encoder,
        ARTIFACT_DIRECTORY / "label_encoder.joblib",
    )

    manifest = {
        "algorithm_id": "xgboost_geochemical_classifier",
        "version": "0.1.0",
        "random_state": RANDOM_STATE,
        "training_samples": int(len(train_indices)),
        "test_samples": int(len(test_indices)),
        "training_deposits": int(len(set(train_deposits))),
        "test_deposits": int(len(set(test_deposits))),
        "classes": label_encoder.classes_.tolist(),
        "elements": element_columns,
        "metrics": metrics,
    }

    with open(
        ARTIFACT_DIRECTORY / "manifest.json",
        "w",
        encoding="utf-8",
    ) as manifest_file:
        json.dump(
            manifest,
            manifest_file,
            indent=2,
        )

    print("\nSaved:")
    print("  artifacts/xgb_model.json")
    print("  artifacts/preprocessing.joblib")
    print("  artifacts/label_encoder.joblib")
    print("  artifacts/manifest.json")


if __name__ == "__main__":
    main()