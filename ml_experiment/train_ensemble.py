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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from train import (
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


MODEL_VERSION = "0.3.0"
CV_RANDOM_STATE = 2027
N_CV_SPLITS = 5

ML_DIRECTORY = Path(__file__).resolve().parent
TUNED_XGB_DIRECTORY = (
    ML_DIRECTORY / "artifacts" / "xgboost_v0_2_0"
)
OUTPUT_DIRECTORY = (
    ML_DIRECTORY / "artifacts" / "ensemble_v0_3_0"
)

SVM_CANDIDATES = [
    {
        "svm_name": "rbf_c1_scale",
        "C": 1.0,
        "gamma": "scale",
    },
    {
        "svm_name": "rbf_c3_scale",
        "C": 3.0,
        "gamma": "scale",
    },
    {
        "svm_name": "rbf_c10_scale",
        "C": 10.0,
        "gamma": "scale",
    },
    {
        "svm_name": "rbf_c3_gamma_0_01",
        "C": 3.0,
        "gamma": 0.01,
    },
]

# xgb_weight=1.0 means XGBoost only.
# xgb_weight=0.0 means SVM only.
XGB_WEIGHTS = [0.0, 0.20, 0.40, 0.50, 0.60, 0.80, 1.0]


def load_locked_experiment() -> tuple[dict, set[str]]:
    manifest_path = TUNED_XGB_DIRECTORY / "manifest.json"
    deposit_path = (
        TUNED_XGB_DIRECTORY / "final_test_deposits.json"
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing tuned manifest: {manifest_path}"
        )

    if not deposit_path.exists():
        raise FileNotFoundError(
            f"Missing locked deposit list: {deposit_path}"
        )

    with open(manifest_path, "r", encoding="utf-8") as file:
        manifest = json.load(file)

    with open(deposit_path, "r", encoding="utf-8") as file:
        final_test_deposits = {
            str(value).strip()
            for value in json.load(file)
        }

    xgb_parameters = dict(manifest["selected_parameters"])
    xgb_parameters.pop("candidate_name", None)

    return xgb_parameters, final_test_deposits


def split_locked_test(
    dataframe: pd.DataFrame,
    final_test_deposits: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    deposit_values = (
        dataframe[GROUP_COLUMN]
        .astype(str)
        .str.strip()
    )

    final_mask = deposit_values.isin(final_test_deposits)

    development_data = (
        dataframe.loc[~final_mask]
        .reset_index(drop=True)
    )
    final_test_data = (
        dataframe.loc[final_mask]
        .reset_index(drop=True)
    )

    missing_deposits = final_test_deposits.difference(
        set(final_test_data[GROUP_COLUMN].astype(str))
    )
    if missing_deposits:
        raise ValueError(
            "Some locked final-test deposits were not found: "
            f"{sorted(missing_deposits)}"
        )

    overlap = set(
        development_data[GROUP_COLUMN].astype(str)
    ).intersection(
        set(final_test_data[GROUP_COLUMN].astype(str))
    )

    if overlap:
        raise RuntimeError(
            f"Deposit leakage detected: {sorted(overlap)}"
        )

    return development_data, final_test_data


def create_xgb_model(
    number_of_classes: int,
    parameters: dict,
) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=number_of_classes,
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        **parameters,
    )


def create_svm_model(parameters: dict) -> SVC:
    return SVC(
        kernel="rbf",
        C=parameters["C"],
        gamma=parameters["gamma"],
        probability=True,
        class_weight=None,
        cache_size=2000,
        random_state=RANDOM_STATE,
    )


def make_clr_features(
    values: np.ndarray,
    censor_mask: np.ndarray,
    missing_mask: np.ndarray,
) -> np.ndarray:
    """
    CLR represents each element relative to the geometric mean of
    all elements in that sample.

    Values must already be positive and imputed.
    """
    natural_logs = np.log(values)
    clr_values = natural_logs - natural_logs.mean(
        axis=1,
        keepdims=True,
    )

    return np.hstack([
        clr_values,
        censor_mask.astype(np.float32),
        missing_mask.astype(np.float32),
    ])


def align_probabilities(
    probabilities: np.ndarray,
    model_classes: np.ndarray,
    number_of_classes: int,
) -> np.ndarray:
    """
    Guarantee that SVM columns use the same encoded-class order as XGBoost.
    """
    aligned = np.zeros(
        (probabilities.shape[0], number_of_classes),
        dtype=float,
    )

    aligned[:, model_classes.astype(int)] = probabilities
    return aligned


def calculate_metrics(
    true_labels: np.ndarray,
    probabilities: np.ndarray,
    number_of_classes: int,
) -> dict[str, float]:
    predictions = np.argmax(probabilities, axis=1)

    return {
        "accuracy": float(
            accuracy_score(true_labels, predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                true_labels,
                predictions,
            )
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


def prepare_features(
    training_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    element_columns: list[str],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    training_values, training_censored, training_missing = (
        prepare_raw_values(
            training_data,
            element_columns,
        )
    )
    validation_values, validation_censored, validation_missing = (
        prepare_raw_values(
            validation_data,
            element_columns,
        )
    )

    medians = fit_training_medians(
        training_values,
        element_columns,
    )

    training_values = apply_medians(
        training_values,
        medians,
    )
    validation_values = apply_medians(
        validation_values,
        medians,
    )

    xgb_training = make_xgboost_features(
        training_values,
        training_censored,
        training_missing,
    )
    xgb_validation = make_xgboost_features(
        validation_values,
        validation_censored,
        validation_missing,
    )

    svm_training_unscaled = make_clr_features(
        training_values,
        training_censored,
        training_missing,
    )
    svm_validation_unscaled = make_clr_features(
        validation_values,
        validation_censored,
        validation_missing,
    )

    scaler = StandardScaler()
    svm_training = scaler.fit_transform(
        svm_training_unscaled
    )
    svm_validation = scaler.transform(
        svm_validation_unscaled
    )

    return (
        xgb_training,
        xgb_validation,
        svm_training,
        svm_validation,
        medians,
    )


def run_cross_validation(
    development_data: pd.DataFrame,
    development_labels: np.ndarray,
    development_groups: np.ndarray,
    element_columns: list[str],
    number_of_classes: int,
    xgb_parameters: dict,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    splitter = StratifiedGroupKFold(
        n_splits=N_CV_SPLITS,
        shuffle=True,
        random_state=CV_RANDOM_STATE,
    )

    result_rows: list[dict] = []

    for fold_number, (
        training_positions,
        validation_positions,
    ) in enumerate(
        splitter.split(
            development_data,
            development_labels,
            groups=development_groups,
        ),
        start=1,
    ):
        print(f"\nFold {fold_number}/{N_CV_SPLITS}")

        training_data = development_data.iloc[
            training_positions
        ]
        validation_data = development_data.iloc[
            validation_positions
        ]

        y_training = development_labels[
            training_positions
        ]
        y_validation = development_labels[
            validation_positions
        ]

        training_deposits = development_groups[
            training_positions
        ]
        validation_deposits = development_groups[
            validation_positions
        ]

        overlap = set(training_deposits).intersection(
            validation_deposits
        )
        if overlap:
            raise RuntimeError(
                f"Deposit leakage in fold {fold_number}: "
                f"{sorted(overlap)}"
            )

        (
            xgb_training,
            xgb_validation,
            svm_training,
            svm_validation,
            _,
        ) = prepare_features(
            training_data,
            validation_data,
            element_columns,
        )

        sample_weights = calculate_training_weights(
            y_training,
            training_deposits,
        )

        xgb_model = create_xgb_model(
            number_of_classes,
            xgb_parameters,
        )
        xgb_model.fit(
            xgb_training,
            y_training,
            sample_weight=sample_weights,
            verbose=False,
        )
        xgb_probabilities = xgb_model.predict_proba(
            xgb_validation
        )

        for svm_parameters in SVM_CANDIDATES:
            svm_name = svm_parameters["svm_name"]
            svm_model = create_svm_model(svm_parameters)

            svm_model.fit(
                svm_training,
                y_training,
                sample_weight=sample_weights,
            )

            raw_svm_probabilities = (
                svm_model.predict_proba(svm_validation)
            )
            svm_probabilities = align_probabilities(
                raw_svm_probabilities,
                svm_model.classes_,
                number_of_classes,
            )

            for xgb_weight in XGB_WEIGHTS:
                combined_probabilities = (
                    xgb_weight * xgb_probabilities
                    + (1.0 - xgb_weight)
                    * svm_probabilities
                )

                metrics = calculate_metrics(
                    y_validation,
                    combined_probabilities,
                    number_of_classes,
                )

                result_rows.append({
                    "fold": fold_number,
                    "svm_name": svm_name,
                    "svm_C": svm_parameters["C"],
                    "svm_gamma": str(
                        svm_parameters["gamma"]
                    ),
                    "xgb_weight": xgb_weight,
                    "svm_weight": 1.0 - xgb_weight,
                    **metrics,
                    "training_samples": len(
                        training_positions
                    ),
                    "validation_samples": len(
                        validation_positions
                    ),
                    "training_deposits": len(
                        set(training_deposits)
                    ),
                    "validation_deposits": len(
                        set(validation_deposits)
                    ),
                })

            svm_only_metrics = calculate_metrics(
                y_validation,
                svm_probabilities,
                number_of_classes,
            )
            print(
                f"  {svm_name}: "
                f"SVM macro_f1="
                f"{svm_only_metrics['macro_f1']:.4f}"
            )

        xgb_only_metrics = calculate_metrics(
            y_validation,
            xgb_probabilities,
            number_of_classes,
        )
        print(
            "  XGBoost only: "
            f"macro_f1={xgb_only_metrics['macro_f1']:.4f}"
        )

    fold_results = pd.DataFrame(result_rows)

    summary = (
        fold_results
        .groupby(
            [
                "svm_name",
                "svm_C",
                "svm_gamma",
                "xgb_weight",
                "svm_weight",
            ],
            as_index=False,
        )
        .agg(
            mean_macro_f1=("macro_f1", "mean"),
            std_macro_f1=("macro_f1", "std"),
            mean_balanced_accuracy=(
                "balanced_accuracy",
                "mean",
            ),
            std_balanced_accuracy=(
                "balanced_accuracy",
                "std",
            ),
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_top_3_accuracy=(
                "top_3_accuracy",
                "mean",
            ),
            std_top_3_accuracy=(
                "top_3_accuracy",
                "std",
            ),
        )
        .sort_values(
            [
                "mean_macro_f1",
                "mean_balanced_accuracy",
                "mean_top_3_accuracy",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    best_row = summary.iloc[0].to_dict()

    selected = {
        "svm_name": best_row["svm_name"],
        "svm_C": float(best_row["svm_C"]),
        "svm_gamma": (
            best_row["svm_gamma"]
            if best_row["svm_gamma"] == "scale"
            else float(best_row["svm_gamma"])
        ),
        "xgb_weight": float(best_row["xgb_weight"]),
        "svm_weight": float(best_row["svm_weight"]),
    }

    return selected, fold_results, summary


def fit_final_ensemble(
    development_data: pd.DataFrame,
    final_test_data: pd.DataFrame,
    development_labels: np.ndarray,
    final_test_labels: np.ndarray,
    development_groups: np.ndarray,
    element_columns: list[str],
    number_of_classes: int,
    xgb_parameters: dict,
    selected: dict,
) -> tuple[
    XGBClassifier,
    SVC,
    StandardScaler,
    np.ndarray,
    np.ndarray,
    dict[str, float],
]:
    development_values, development_censored, development_missing = (
        prepare_raw_values(
            development_data,
            element_columns,
        )
    )
    test_values, test_censored, test_missing = (
        prepare_raw_values(
            final_test_data,
            element_columns,
        )
    )

    medians = fit_training_medians(
        development_values,
        element_columns,
    )

    development_values = apply_medians(
        development_values,
        medians,
    )
    test_values = apply_medians(
        test_values,
        medians,
    )

    xgb_development = make_xgboost_features(
        development_values,
        development_censored,
        development_missing,
    )
    xgb_test = make_xgboost_features(
        test_values,
        test_censored,
        test_missing,
    )

    svm_development_unscaled = make_clr_features(
        development_values,
        development_censored,
        development_missing,
    )
    svm_test_unscaled = make_clr_features(
        test_values,
        test_censored,
        test_missing,
    )

    scaler = StandardScaler()
    svm_development = scaler.fit_transform(
        svm_development_unscaled
    )
    svm_test = scaler.transform(
        svm_test_unscaled
    )

    sample_weights = calculate_training_weights(
        development_labels,
        development_groups,
    )

    xgb_model = create_xgb_model(
        number_of_classes,
        xgb_parameters,
    )
    xgb_model.fit(
        xgb_development,
        development_labels,
        sample_weight=sample_weights,
        verbose=False,
    )

    svm_model = SVC(
        kernel="rbf",
        C=selected["svm_C"],
        gamma=selected["svm_gamma"],
        probability=True,
        class_weight=None,
        cache_size=2000,
        random_state=RANDOM_STATE,
    )
    svm_model.fit(
        svm_development,
        development_labels,
        sample_weight=sample_weights,
    )

    xgb_probabilities = xgb_model.predict_proba(xgb_test)

    svm_probabilities = align_probabilities(
        svm_model.predict_proba(svm_test),
        svm_model.classes_,
        number_of_classes,
    )

    combined_probabilities = (
        selected["xgb_weight"] * xgb_probabilities
        + selected["svm_weight"] * svm_probabilities
    )

    final_metrics = calculate_metrics(
        final_test_labels,
        combined_probabilities,
        number_of_classes,
    )

    return (
        xgb_model,
        svm_model,
        scaler,
        medians,
        combined_probabilities,
        final_metrics,
    )


def save_predictions(
    final_test_data: pd.DataFrame,
    true_labels: np.ndarray,
    probabilities: np.ndarray,
    label_encoder: LabelEncoder,
    output_path: Path,
) -> None:
    predictions = np.argmax(probabilities, axis=1)
    top_indices = np.argsort(
        probabilities,
        axis=1,
    )[:, ::-1][:, :3]

    output = pd.DataFrame({
        "deposit_id": (
            final_test_data[GROUP_COLUMN]
            .astype(str)
            .to_numpy()
        ),
        "true_class": label_encoder.inverse_transform(
            true_labels
        ),
        "predicted_class": label_encoder.inverse_transform(
            predictions
        ),
        "top_1_class": label_encoder.inverse_transform(
            top_indices[:, 0]
        ),
        "top_1_probability": probabilities[
            np.arange(len(probabilities)),
            top_indices[:, 0],
        ],
        "top_2_class": label_encoder.inverse_transform(
            top_indices[:, 1]
        ),
        "top_2_probability": probabilities[
            np.arange(len(probabilities)),
            top_indices[:, 1],
        ],
        "top_3_class": label_encoder.inverse_transform(
            top_indices[:, 2]
        ),
        "top_3_probability": probabilities[
            np.arange(len(probabilities)),
            top_indices[:, 2],
        ],
    })

    for possible_id_column in [
        "Sample",
        "SAMPLE CODE",
        "Deposit Name",
    ]:
        if possible_id_column in final_test_data.columns:
            output.insert(
                0,
                possible_id_column,
                final_test_data[
                    possible_id_column
                ].astype(str).to_numpy(),
            )

    output.to_csv(output_path, index=False)


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    xgb_parameters, locked_test_deposits = (
        load_locked_experiment()
    )

    dataframe = load_osnaca()
    dataframe = clean_labels(dataframe)
    element_columns = identify_element_columns(dataframe)

    development_data, final_test_data = split_locked_test(
        dataframe,
        locked_test_deposits,
    )

    label_encoder = LabelEncoder()
    label_encoder.fit(dataframe[TARGET_COLUMN])

    development_labels = label_encoder.transform(
        development_data[TARGET_COLUMN]
    )
    final_test_labels = label_encoder.transform(
        final_test_data[TARGET_COLUMN]
    )

    development_groups = (
        development_data[GROUP_COLUMN]
        .astype(str)
        .to_numpy()
    )

    number_of_classes = len(label_encoder.classes_)

    print("\nData split:")
    print(
        f"Development samples: {len(development_data):,}"
    )
    print(
        f"Development deposits: "
        f"{development_data[GROUP_COLUMN].nunique():,}"
    )
    print(
        f"Locked test samples: {len(final_test_data):,}"
    )
    print(
        f"Locked test deposits: "
        f"{final_test_data[GROUP_COLUMN].nunique():,}"
    )

    print(
        "\nRunning grouped cross-validation for "
        "SVM settings and ensemble weights..."
    )

    selected, fold_results, summary = (
        run_cross_validation(
            development_data=development_data,
            development_labels=development_labels,
            development_groups=development_groups,
            element_columns=element_columns,
            number_of_classes=number_of_classes,
            xgb_parameters=xgb_parameters,
        )
    )

    fold_results.to_csv(
        OUTPUT_DIRECTORY / "cv_fold_results.csv",
        index=False,
    )
    summary.to_csv(
        OUTPUT_DIRECTORY / "cv_summary.csv",
        index=False,
    )

    print("\nTop cross-validation combinations:")
    print(
        summary.head(10)
        .round(4)
        .to_string(index=False)
    )

    print("\nSelected ensemble:")
    print(json.dumps(selected, indent=2))

    (
        xgb_model,
        svm_model,
        scaler,
        medians,
        final_probabilities,
        final_metrics,
    ) = fit_final_ensemble(
        development_data=development_data,
        final_test_data=final_test_data,
        development_labels=development_labels,
        final_test_labels=final_test_labels,
        development_groups=development_groups,
        element_columns=element_columns,
        number_of_classes=number_of_classes,
        xgb_parameters=xgb_parameters,
        selected=selected,
    )

    print("\nLocked final-test ensemble metrics:")
    for name, value in final_metrics.items():
        print(f"{name}: {value:.4f}")

    final_predictions = np.argmax(
        final_probabilities,
        axis=1,
    )

    report = classification_report(
        final_test_labels,
        final_predictions,
        labels=np.arange(number_of_classes),
        target_names=label_encoder.classes_,
        zero_division=0,
        output_dict=True,
    )

    pd.DataFrame(report).transpose().to_csv(
        OUTPUT_DIRECTORY
        / "final_classification_report.csv"
    )

    matrix = confusion_matrix(
        final_test_labels,
        final_predictions,
        labels=np.arange(number_of_classes),
        normalize="true",
    )

    pd.DataFrame(
        matrix,
        index=label_encoder.classes_,
        columns=label_encoder.classes_,
    ).to_csv(
        OUTPUT_DIRECTORY / "final_confusion_matrix.csv"
    )

    save_predictions(
        final_test_data,
        final_test_labels,
        final_probabilities,
        label_encoder,
        OUTPUT_DIRECTORY / "final_test_predictions.csv",
    )

    xgb_model.save_model(
        str(OUTPUT_DIRECTORY / "xgb_model.json")
    )
    joblib.dump(
        svm_model,
        OUTPUT_DIRECTORY / "svm_model.joblib",
    )
    joblib.dump(
        label_encoder,
        OUTPUT_DIRECTORY / "label_encoder.joblib",
    )

    preprocessing = {
        "element_columns": element_columns,
        "medians": medians,
        "svm_scaler": scaler,
        "xgb_feature_representation": (
            "log10_values_with_censor_and_missing_masks"
        ),
        "svm_feature_representation": (
            "clr_values_with_censor_and_missing_masks"
        ),
        "censored_policy": (
            "absolute_detection_limit_divided_by_2"
        ),
    }
    joblib.dump(
        preprocessing,
        OUTPUT_DIRECTORY / "preprocessing.joblib",
    )

    manifest = {
        "algorithm_id": (
            "xgboost_rbf_svm_geochemical_ensemble"
        ),
        "version": MODEL_VERSION,
        "random_state": RANDOM_STATE,
        "cv_random_state": CV_RANDOM_STATE,
        "development_samples": len(development_data),
        "final_test_samples": len(final_test_data),
        "development_deposits": int(
            development_data[GROUP_COLUMN].nunique()
        ),
        "final_test_deposits": int(
            final_test_data[GROUP_COLUMN].nunique()
        ),
        "classes": label_encoder.classes_.tolist(),
        "elements": element_columns,
        "xgb_parameters": xgb_parameters,
        "selected_ensemble": selected,
        "selection_metric": "mean_macro_f1",
        "cross_validation": summary.iloc[0].to_dict(),
        "final_test_metrics": final_metrics,
        "notes": (
            "SVM settings and ensemble weights were selected "
            "only from deposit-grouped cross-validation on the "
            "development set. The locked test deposits match "
            "xgboost v0.2.0."
        ),
    }

    with open(
        OUTPUT_DIRECTORY / "manifest.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(manifest, file, indent=2)

    print(
        f"\nSaved ensemble artifacts to:\n"
        f"{OUTPUT_DIRECTORY}"
    )


if __name__ == "__main__":
    main()
