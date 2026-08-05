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
    confusion_matrix,
    f1_score,
    top_k_accuracy_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

# Reuse the data loading and preprocessing functions that already work
# in your existing ml_experiment/train.py.
from train import (
    ARTIFACT_DIRECTORY as BASE_ARTIFACT_DIRECTORY,
    GROUP_COLUMN,
    RANDOM_STATE,
    TARGET_COLUMN,
    apply_medians,
    calculate_training_weights,
    clean_labels,
    fit_training_medians,
    identify_element_columns,
    load_osnaca,
    make_xgboost_features,
    prepare_raw_values,
)


MODEL_VERSION = "0.2.0"
FINAL_TEST_RANDOM_STATE = 2026
CV_RANDOM_STATE = 2027
N_CV_SPLITS = 5

OUTPUT_DIRECTORY = (
    Path(BASE_ARTIFACT_DIRECTORY) / f"xgboost_v{MODEL_VERSION.replace('.', '_')}"
)

# A small, deliberate search is easier to audit than a huge parameter grid.
# The first entry resembles the initial baseline; the others apply lower
# learning rates and stronger regularisation.
PARAMETER_CANDIDATES = [
    {
        "candidate_name": "baseline_like",
        "n_estimators": 700,
        "learning_rate": 0.30,
        "max_depth": 6,
        "min_child_weight": 1,
        "subsample": 1.00,
        "colsample_bytree": 1.00,
        "reg_alpha": 0.00,
        "reg_lambda": 1.00,
        "gamma": 0.00,
    },
    {
        "candidate_name": "regularised_depth_3",
        "n_estimators": 900,
        "learning_rate": 0.04,
        "max_depth": 3,
        "min_child_weight": 2,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.05,
        "reg_lambda": 4.00,
        "gamma": 0.00,
    },
    {
        "candidate_name": "regularised_depth_4",
        "n_estimators": 900,
        "learning_rate": 0.04,
        "max_depth": 4,
        "min_child_weight": 2,
        "subsample": 0.85,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.10,
        "reg_lambda": 6.00,
        "gamma": 0.05,
    },
    {
        "candidate_name": "slow_depth_4",
        "n_estimators": 1200,
        "learning_rate": 0.025,
        "max_depth": 4,
        "min_child_weight": 3,
        "subsample": 0.80,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.10,
        "reg_lambda": 8.00,
        "gamma": 0.05,
    },
    {
        "candidate_name": "slow_depth_5",
        "n_estimators": 1200,
        "learning_rate": 0.025,
        "max_depth": 5,
        "min_child_weight": 3,
        "subsample": 0.80,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.20,
        "reg_lambda": 10.00,
        "gamma": 0.10,
    },
    {
        "candidate_name": "strong_regularisation",
        "n_estimators": 1400,
        "learning_rate": 0.02,
        "max_depth": 4,
        "min_child_weight": 5,
        "subsample": 0.75,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.50,
        "reg_lambda": 15.00,
        "gamma": 0.10,
    },
]


def create_model(
    number_of_classes: int,
    parameters: dict,
) -> XGBClassifier:
    model_parameters = {
        key: value
        for key, value in parameters.items()
        if key != "candidate_name"
    }

    return XGBClassifier(
        objective="multi:softprob",
        num_class=number_of_classes,
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        **model_parameters,
    )


def make_fresh_final_test_split(
    dataframe: pd.DataFrame,
    encoded_labels: np.ndarray,
    deposit_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct the original split.

    The original 200-sample holdout becomes development/validation data.
    A fresh final test fold is selected only from the original training
    deposits, so its deposits were not in the test results already examined.
    """
    original_splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    original_training_indices, original_holdout_indices = next(
        original_splitter.split(
            dataframe,
            encoded_labels,
            groups=deposit_ids,
        )
    )

    original_training_data = dataframe.iloc[original_training_indices]
    original_training_labels = encoded_labels[original_training_indices]
    original_training_groups = deposit_ids[original_training_indices]

    fresh_splitter = StratifiedGroupKFold(
        n_splits=4,
        shuffle=True,
        random_state=FINAL_TEST_RANDOM_STATE,
    )

    _, final_test_positions = next(
        fresh_splitter.split(
            original_training_data,
            original_training_labels,
            groups=original_training_groups,
        )
    )

    final_test_indices = original_training_indices[final_test_positions]

    all_indices = np.arange(len(dataframe))
    development_indices = np.setdiff1d(
        all_indices,
        final_test_indices,
        assume_unique=False,
    )

    development_groups = set(deposit_ids[development_indices])
    final_test_groups = set(deposit_ids[final_test_indices])

    overlap = development_groups.intersection(final_test_groups)
    if overlap:
        raise RuntimeError(
            f"Deposit leakage detected in final split: {sorted(overlap)}"
        )

    # Confirm that the fresh test deposits were not in the original holdout.
    old_holdout_groups = set(deposit_ids[original_holdout_indices])
    reused_groups = old_holdout_groups.intersection(final_test_groups)
    if reused_groups:
        raise RuntimeError(
            "Fresh final test unexpectedly overlaps the original holdout: "
            f"{sorted(reused_groups)}"
        )

    return (
        development_indices,
        final_test_indices,
        original_holdout_indices,
    )


def prepare_fold_features(
    training_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    element_columns: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    training_values, training_censored, training_missing = (
        prepare_raw_values(training_data, element_columns)
    )
    validation_values, validation_censored, validation_missing = (
        prepare_raw_values(validation_data, element_columns)
    )

    medians = fit_training_medians(
        training_values,
        element_columns,
    )

    training_values = apply_medians(training_values, medians)
    validation_values = apply_medians(validation_values, medians)

    X_training = make_xgboost_features(
        training_values,
        training_censored,
        training_missing,
    )
    X_validation = make_xgboost_features(
        validation_values,
        validation_censored,
        validation_missing,
    )

    return X_training, X_validation, medians


def calculate_metrics(
    true_labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    number_of_classes: int,
) -> dict[str, float]:
    return {
        "accuracy": float(
            accuracy_score(true_labels, predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(true_labels, predictions)
        ),
        "macro_f1": float(
            f1_score(
                true_labels,
                predictions,
                average="macro",
            )
        ),
        "top_3_accuracy": float(
            top_k_accuracy_score(
                true_labels,
                probabilities,
                k=min(3, number_of_classes),
                labels=np.arange(number_of_classes),
            )
        ),
    }


def tune_parameters(
    development_data: pd.DataFrame,
    development_labels: np.ndarray,
    development_groups: np.ndarray,
    element_columns: list[str],
    number_of_classes: int,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    splitter = StratifiedGroupKFold(
        n_splits=N_CV_SPLITS,
        shuffle=True,
        random_state=CV_RANDOM_STATE,
    )

    fold_rows: list[dict] = []

    for candidate_number, parameters in enumerate(
        PARAMETER_CANDIDATES,
        start=1,
    ):
        candidate_name = parameters["candidate_name"]
        print(
            f"\nCandidate {candidate_number}/"
            f"{len(PARAMETER_CANDIDATES)}: {candidate_name}"
        )

        for fold_number, (training_positions, validation_positions) in enumerate(
            splitter.split(
                development_data,
                development_labels,
                groups=development_groups,
            ),
            start=1,
        ):
            training_data = development_data.iloc[training_positions]
            validation_data = development_data.iloc[validation_positions]

            y_training = development_labels[training_positions]
            y_validation = development_labels[validation_positions]

            training_deposits = development_groups[training_positions]
            validation_deposits = development_groups[validation_positions]

            overlap = set(training_deposits).intersection(validation_deposits)
            if overlap:
                raise RuntimeError(
                    f"Deposit leakage in CV fold {fold_number}: "
                    f"{sorted(overlap)}"
                )

            missing_training_classes = (
                set(range(number_of_classes))
                - set(np.unique(y_training))
            )
            if missing_training_classes:
                raise RuntimeError(
                    "A training fold is missing encoded classes: "
                    f"{sorted(missing_training_classes)}"
                )

            X_training, X_validation, _ = prepare_fold_features(
                training_data,
                validation_data,
                element_columns,
            )

            training_weights = calculate_training_weights(
                y_training,
                training_deposits,
            )

            model = create_model(
                number_of_classes=number_of_classes,
                parameters=parameters,
            )

            model.fit(
                X_training,
                y_training,
                sample_weight=training_weights,
                verbose=False,
            )

            predictions = model.predict(X_validation)
            probabilities = model.predict_proba(X_validation)

            metrics = calculate_metrics(
                y_validation,
                predictions,
                probabilities,
                number_of_classes,
            )

            fold_row = {
                "candidate_name": candidate_name,
                "fold": fold_number,
                **{
                    key: value
                    for key, value in parameters.items()
                    if key != "candidate_name"
                },
                **metrics,
                "training_samples": len(training_positions),
                "validation_samples": len(validation_positions),
                "training_deposits": len(set(training_deposits)),
                "validation_deposits": len(set(validation_deposits)),
            }
            fold_rows.append(fold_row)

            print(
                f"  Fold {fold_number}: "
                f"macro_f1={metrics['macro_f1']:.4f}, "
                f"balanced_accuracy="
                f"{metrics['balanced_accuracy']:.4f}, "
                f"top_3={metrics['top_3_accuracy']:.4f}"
            )

    fold_results = pd.DataFrame(fold_rows)

    summary = (
        fold_results
        .groupby("candidate_name", as_index=False)
        .agg(
            mean_macro_f1=("macro_f1", "mean"),
            std_macro_f1=("macro_f1", "std"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            std_balanced_accuracy=("balanced_accuracy", "std"),
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_top_3_accuracy=("top_3_accuracy", "mean"),
            std_top_3_accuracy=("top_3_accuracy", "std"),
        )
        .sort_values(
            ["mean_macro_f1", "mean_balanced_accuracy"],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    best_candidate_name = summary.iloc[0]["candidate_name"]

    best_parameters = next(
        parameters
        for parameters in PARAMETER_CANDIDATES
        if parameters["candidate_name"] == best_candidate_name
    )

    return best_parameters, fold_results, summary


def save_test_predictions(
    test_data: pd.DataFrame,
    true_labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    label_encoder,
    output_path: Path,
) -> None:
    best_indices = np.argsort(probabilities, axis=1)[:, ::-1][:, :3]

    output = pd.DataFrame({
        "deposit_id": test_data[GROUP_COLUMN].astype(str).to_numpy(),
        "true_class": label_encoder.inverse_transform(true_labels),
        "predicted_class": label_encoder.inverse_transform(predictions),
        "top_1_class": label_encoder.inverse_transform(best_indices[:, 0]),
        "top_1_probability": probabilities[
            np.arange(len(probabilities)),
            best_indices[:, 0],
        ],
        "top_2_class": label_encoder.inverse_transform(best_indices[:, 1]),
        "top_2_probability": probabilities[
            np.arange(len(probabilities)),
            best_indices[:, 1],
        ],
        "top_3_class": label_encoder.inverse_transform(best_indices[:, 2]),
        "top_3_probability": probabilities[
            np.arange(len(probabilities)),
            best_indices[:, 2],
        ],
    })

    for possible_id_column in ["Sample", "SAMPLE CODE", "Deposit Name"]:
        if possible_id_column in test_data.columns:
            output.insert(
                0,
                possible_id_column,
                test_data[possible_id_column].astype(str).to_numpy(),
            )

    output.to_csv(output_path, index=False)


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    dataframe = load_osnaca()
    dataframe = clean_labels(dataframe)
    element_columns = identify_element_columns(dataframe)

    labels = dataframe[TARGET_COLUMN].to_numpy()
    deposit_ids = dataframe[GROUP_COLUMN].to_numpy()

    # Use a fresh label encoder for this model version.
    from sklearn.preprocessing import LabelEncoder

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    number_of_classes = len(label_encoder.classes_)

    (
        development_indices,
        final_test_indices,
        original_holdout_indices,
    ) = make_fresh_final_test_split(
        dataframe,
        encoded_labels,
        deposit_ids,
    )

    development_data = dataframe.iloc[development_indices].reset_index(drop=True)
    final_test_data = dataframe.iloc[final_test_indices].reset_index(drop=True)

    development_labels = encoded_labels[development_indices]
    final_test_labels = encoded_labels[final_test_indices]

    development_groups = deposit_ids[development_indices]
    final_test_groups = deposit_ids[final_test_indices]

    print("\nData roles:")
    print(
        f"Original examined holdout samples now available for "
        f"development: {len(original_holdout_indices):,}"
    )
    print(f"Development samples: {len(development_data):,}")
    print(
        f"Development deposits: "
        f"{len(set(development_groups)):,}"
    )
    print(f"Fresh final test samples: {len(final_test_data):,}")
    print(
        f"Fresh final test deposits: "
        f"{len(set(final_test_groups)):,}"
    )

    print("\nStarting deposit-grouped cross-validation tuning...")

    (
        best_parameters,
        fold_results,
        cv_summary,
    ) = tune_parameters(
        development_data=development_data,
        development_labels=development_labels,
        development_groups=development_groups,
        element_columns=element_columns,
        number_of_classes=number_of_classes,
    )

    fold_results.to_csv(
        OUTPUT_DIRECTORY / "cv_fold_results.csv",
        index=False,
    )
    cv_summary.to_csv(
        OUTPUT_DIRECTORY / "cv_summary.csv",
        index=False,
    )

    print("\nCross-validation summary:")
    print(cv_summary.round(4).to_string(index=False))

    print("\nSelected parameters:")
    print(json.dumps(best_parameters, indent=2))

    # Fit preprocessing using all development data only.
    development_values, development_censored, development_missing = (
        prepare_raw_values(development_data, element_columns)
    )
    final_test_values, final_test_censored, final_test_missing = (
        prepare_raw_values(final_test_data, element_columns)
    )

    medians = fit_training_medians(
        development_values,
        element_columns,
    )

    development_values = apply_medians(
        development_values,
        medians,
    )
    final_test_values = apply_medians(
        final_test_values,
        medians,
    )

    X_development = make_xgboost_features(
        development_values,
        development_censored,
        development_missing,
    )
    X_final_test = make_xgboost_features(
        final_test_values,
        final_test_censored,
        final_test_missing,
    )

    development_weights = calculate_training_weights(
        development_labels,
        development_groups,
    )

    final_model = create_model(
        number_of_classes=number_of_classes,
        parameters=best_parameters,
    )

    print("\nTraining selected model on all development deposits...")
    final_model.fit(
        X_development,
        development_labels,
        sample_weight=development_weights,
        verbose=False,
    )

    print("\nEvaluating once on the fresh final test deposits...")
    final_predictions = final_model.predict(X_final_test)
    final_probabilities = final_model.predict_proba(X_final_test)

    final_metrics = calculate_metrics(
        final_test_labels,
        final_predictions,
        final_probabilities,
        number_of_classes,
    )

    print("\nFresh final test metrics:")
    for metric_name, value in final_metrics.items():
        print(f"{metric_name}: {value:.4f}")

    report = classification_report(
        final_test_labels,
        final_predictions,
        labels=np.arange(number_of_classes),
        target_names=label_encoder.classes_,
        zero_division=0,
        output_dict=True,
    )

    report_dataframe = pd.DataFrame(report).transpose()
    report_dataframe.to_csv(
        OUTPUT_DIRECTORY / "final_classification_report.csv"
    )

    normalised_confusion = confusion_matrix(
        final_test_labels,
        final_predictions,
        labels=np.arange(number_of_classes),
        normalize="true",
    )

    confusion_dataframe = pd.DataFrame(
        normalised_confusion,
        index=label_encoder.classes_,
        columns=label_encoder.classes_,
    )
    confusion_dataframe.to_csv(
        OUTPUT_DIRECTORY / "final_confusion_matrix.csv"
    )

    save_test_predictions(
        test_data=final_test_data,
        true_labels=final_test_labels,
        predictions=final_predictions,
        probabilities=final_probabilities,
        label_encoder=label_encoder,
        output_path=OUTPUT_DIRECTORY / "final_test_predictions.csv",
    )

    final_model.save_model(
        str(OUTPUT_DIRECTORY / "xgb_model.json")
    )

    preprocessing_artifact = {
        "element_columns": element_columns,
        "medians": medians,
        "target_column": TARGET_COLUMN,
        "group_column": GROUP_COLUMN,
        "censored_policy": (
            "absolute_detection_limit_divided_by_2"
        ),
        "feature_representation": (
            "log10_values_with_censor_and_missing_masks"
        ),
    }

    joblib.dump(
        preprocessing_artifact,
        OUTPUT_DIRECTORY / "preprocessing.joblib",
    )
    joblib.dump(
        label_encoder,
        OUTPUT_DIRECTORY / "label_encoder.joblib",
    )

    final_test_deposits = sorted(
        str(deposit)
        for deposit in set(final_test_groups)
    )
    with open(
        OUTPUT_DIRECTORY / "final_test_deposits.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(final_test_deposits, file, indent=2)

    manifest = {
        "algorithm_id": "xgboost_geochemical_classifier",
        "version": MODEL_VERSION,
        "random_state": RANDOM_STATE,
        "final_test_random_state": FINAL_TEST_RANDOM_STATE,
        "cv_random_state": CV_RANDOM_STATE,
        "development_samples": int(len(development_data)),
        "final_test_samples": int(len(final_test_data)),
        "development_deposits": int(len(set(development_groups))),
        "final_test_deposits": int(len(set(final_test_groups))),
        "classes": label_encoder.classes_.tolist(),
        "elements": element_columns,
        "selected_parameters": best_parameters,
        "selection_metric": "mean_macro_f1",
        "cross_validation": cv_summary.iloc[0].to_dict(),
        "final_test_metrics": final_metrics,
        "notes": (
            "The original 200-sample holdout was treated as development "
            "data. The final test deposits were selected from the original "
            "training partition and evaluated only after parameter selection."
        ),
    }

    with open(
        OUTPUT_DIRECTORY / "manifest.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(manifest, file, indent=2)

    print(f"\nSaved tuned model and reports to:\n{OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()
