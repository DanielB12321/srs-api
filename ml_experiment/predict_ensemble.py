from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier


ML_DIRECTORY = Path(__file__).resolve().parent
ARTIFACT_DIRECTORY = (
    ML_DIRECTORY / "artifacts" / "ensemble_v0_3_0"
)


def load_artifacts():
    required_paths = {
        "xgboost": ARTIFACT_DIRECTORY / "xgb_model.json",
        "svm": ARTIFACT_DIRECTORY / "svm_model.joblib",
        "preprocessing": ARTIFACT_DIRECTORY / "preprocessing.joblib",
        "labels": ARTIFACT_DIRECTORY / "label_encoder.joblib",
        "manifest": ARTIFACT_DIRECTORY / "manifest.json",
    }

    for name, path in required_paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {name} artifact: {path}"
            )

    xgb_model = XGBClassifier()
    xgb_model.load_model(required_paths["xgboost"])

    svm_model = joblib.load(required_paths["svm"])
    preprocessing = joblib.load(
        required_paths["preprocessing"]
    )
    label_encoder = joblib.load(required_paths["labels"])

    with open(
        required_paths["manifest"],
        "r",
        encoding="utf-8",
    ) as file:
        manifest = json.load(file)

    return (
        xgb_model,
        svm_model,
        preprocessing,
        label_encoder,
        manifest,
    )


def load_input(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Input file does not exist: {path}"
        )

    suffix = path.suffix.lower()

    if suffix == ".csv":
        dataframe = pd.read_csv(path)
    elif suffix in {".xlsx", ".xls"}:
        dataframe = pd.read_excel(path)
    else:
        raise ValueError(
            "Input file must be CSV or Excel."
        )

    dataframe.columns = (
        dataframe.columns.astype(str).str.strip()
    )

    if dataframe.empty:
        raise ValueError("Input dataset is empty.")

    return dataframe


def prepare_values(
    dataframe: pd.DataFrame,
    element_columns: list[str],
    medians: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    missing_elements = [
        element
        for element in element_columns
        if element not in dataframe.columns
    ]

    available_elements = [
        element
        for element in element_columns
        if element in dataframe.columns
    ]

    if not available_elements:
        raise ValueError(
            "The input dataset contains none of the "
            "elements expected by the model."
        )

    aligned = dataframe.reindex(columns=element_columns)

    for column in element_columns:
        aligned[column] = pd.to_numeric(
            aligned[column],
            errors="coerce",
        )

    values = aligned.to_numpy(
        dtype=float,
        copy=True,
    )

    censor_mask = np.isfinite(values) & (values < 0)
    missing_mask = ~np.isfinite(values)

    # Same below-detection policy used during training.
    values[censor_mask] = (
        np.abs(values[censor_mask]) / 2.0
    )

    values[
        np.isfinite(values) & (values <= 0)
    ] = np.nan

    missing_rows, missing_columns = np.where(
        ~np.isfinite(values)
    )
    values[missing_rows, missing_columns] = (
        medians[missing_columns]
    )

    values = np.clip(values, 1e-12, None)

    return (
        values,
        censor_mask,
        missing_mask,
        missing_elements,
    )


def make_xgb_features(
    values: np.ndarray,
    censor_mask: np.ndarray,
    missing_mask: np.ndarray,
) -> np.ndarray:
    return np.hstack([
        np.log10(values),
        censor_mask.astype(np.float32),
        missing_mask.astype(np.float32),
    ])


def make_svm_features(
    values: np.ndarray,
    censor_mask: np.ndarray,
    missing_mask: np.ndarray,
    scaler,
) -> np.ndarray:
    natural_logs = np.log(values)

    clr_values = (
        natural_logs
        - natural_logs.mean(axis=1, keepdims=True)
    )

    unscaled_features = np.hstack([
        clr_values,
        censor_mask.astype(np.float32),
        missing_mask.astype(np.float32),
    ])

    return scaler.transform(unscaled_features)


def align_probabilities(
    probabilities: np.ndarray,
    model_classes: np.ndarray,
    number_of_classes: int,
) -> np.ndarray:
    aligned = np.zeros(
        (probabilities.shape[0], number_of_classes),
        dtype=float,
    )

    aligned[:, model_classes.astype(int)] = probabilities

    return aligned


def create_prediction_results(
    dataframe: pd.DataFrame,
    probabilities: np.ndarray,
    label_encoder,
    top_k: int = 3,
) -> pd.DataFrame:
    top_k = min(top_k, probabilities.shape[1])

    top_indices = np.argsort(
        probabilities,
        axis=1,
    )[:, ::-1][:, :top_k]

    results = pd.DataFrame({
        "sample_index": np.arange(len(dataframe)),
        "predicted_class": (
            label_encoder.inverse_transform(
                top_indices[:, 0]
            )
        ),
        "confidence": probabilities[
            np.arange(len(dataframe)),
            top_indices[:, 0],
        ],
    })

    for rank in range(top_k):
        class_indices = top_indices[:, rank]

        results[f"top_{rank + 1}_class"] = (
            label_encoder.inverse_transform(
                class_indices
            )
        )

        results[f"top_{rank + 1}_probability"] = (
            probabilities[
                np.arange(len(dataframe)),
                class_indices,
            ]
        )

    # Preserve useful identifiers from the input.
    identifier_columns = [
        "Sample",
        "SAMPLE CODE",
        "Deposit Name",
        "Three Character Code",
    ]

    for column in reversed(identifier_columns):
        if column in dataframe.columns:
            results.insert(
                0,
                column,
                dataframe[column].to_numpy(),
            )

    return results


def predict(
    dataframe: pd.DataFrame,
    xgb_model,
    svm_model,
    preprocessing: dict,
    label_encoder,
    manifest: dict,
) -> tuple[pd.DataFrame, list[str]]:
    element_columns = preprocessing["element_columns"]
    medians = np.asarray(
        preprocessing["medians"],
        dtype=float,
    )
    scaler = preprocessing["svm_scaler"]

    (
        values,
        censor_mask,
        missing_mask,
        missing_elements,
    ) = prepare_values(
        dataframe,
        element_columns,
        medians,
    )

    xgb_features = make_xgb_features(
        values,
        censor_mask,
        missing_mask,
    )

    svm_features = make_svm_features(
        values,
        censor_mask,
        missing_mask,
        scaler,
    )

    xgb_probabilities = xgb_model.predict_proba(
        xgb_features
    )

    raw_svm_probabilities = svm_model.predict_proba(
        svm_features
    )

    svm_probabilities = align_probabilities(
        raw_svm_probabilities,
        svm_model.classes_,
        len(label_encoder.classes_),
    )

    ensemble = manifest["selected_ensemble"]

    xgb_weight = float(ensemble["xgb_weight"])
    svm_weight = float(ensemble["svm_weight"])

    # Soft-voting probability combination.
    combined_probabilities = (
        xgb_weight * xgb_probabilities
        + svm_weight * svm_probabilities
    )

    # Protect against tiny floating-point drift.
    combined_probabilities = (
        combined_probabilities
        / combined_probabilities.sum(
            axis=1,
            keepdims=True,
        )
    )

    results = create_prediction_results(
        dataframe,
        combined_probabilities,
        label_encoder,
        top_k=3,
    )

    return results, missing_elements


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the XGBoost-RBF SVM geochemical "
            "ensemble on a CSV or Excel dataset."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV or Excel path.",
    )

    parser.add_argument(
        "--output",
        default=(
            "ml_experiment/"
            "ensemble_prediction_results.csv"
        ),
        help="Output CSV path.",
    )

    args = parser.parse_args()

    (
        xgb_model,
        svm_model,
        preprocessing,
        label_encoder,
        manifest,
    ) = load_artifacts()

    dataframe = load_input(Path(args.input))

    results, missing_elements = predict(
        dataframe,
        xgb_model,
        svm_model,
        preprocessing,
        label_encoder,
        manifest,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(output_path, index=False)

    ensemble = manifest["selected_ensemble"]

    print("\nEnsemble weights:")
    print(
        f"XGBoost: "
        f"{float(ensemble['xgb_weight']):.2f}"
    )
    print(
        f"RBF SVM: "
        f"{float(ensemble['svm_weight']):.2f}"
    )

    if missing_elements:
        print(
            "\nWarning: these expected elements were "
            "not present and were imputed:"
        )
        print(", ".join(missing_elements))

    print("\nPredictions:")
    print(results.to_string(index=False))

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
