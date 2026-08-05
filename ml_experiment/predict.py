from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier


ML_DIRECTORY = Path(__file__).resolve().parent
ARTIFACT_DIRECTORY = ML_DIRECTORY / "artifacts"


def load_artifacts():
    model_path = ARTIFACT_DIRECTORY / "xgb_model.json"
    preprocessing_path = (
        ARTIFACT_DIRECTORY / "preprocessing.joblib"
    )
    label_encoder_path = (
        ARTIFACT_DIRECTORY / "label_encoder.joblib"
    )

    required_files = [
        model_path,
        preprocessing_path,
        label_encoder_path,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing model artifact: {path}"
            )

    model = XGBClassifier()
    model.load_model(model_path)

    preprocessing = joblib.load(preprocessing_path)
    label_encoder = joblib.load(label_encoder_path)

    return model, preprocessing, label_encoder


def load_uploaded_data(path: Path) -> pd.DataFrame:
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
            "Input must be a CSV or Excel file."
        )

    dataframe.columns = (
        dataframe.columns
        .astype(str)
        .str.strip()
    )

    if dataframe.empty:
        raise ValueError("The uploaded dataset is empty.")

    return dataframe


def create_features(
    dataframe: pd.DataFrame,
    preprocessing: dict,
) -> tuple[np.ndarray, list[str]]:
    expected_elements = preprocessing["element_columns"]
    medians = np.asarray(preprocessing["medians"])

    missing_elements = [
        element
        for element in expected_elements
        if element not in dataframe.columns
    ]

    available_elements = [
        element
        for element in expected_elements
        if element in dataframe.columns
    ]

    if not available_elements:
        raise ValueError(
            "The uploaded dataset does not contain any "
            "elements expected by the trained model."
        )

    # Reindex creates NaN columns for elements absent from the upload.
    aligned = dataframe.reindex(columns=expected_elements)

    for column in expected_elements:
        aligned[column] = pd.to_numeric(
            aligned[column],
            errors="coerce",
        )

    values = aligned.to_numpy(dtype=float, copy=True)

    censor_mask = np.isfinite(values) & (values < 0)
    missing_mask = ~np.isfinite(values)

    # Same detection-limit handling used during training.
    values[censor_mask] = (
        np.abs(values[censor_mask]) / 2.0
    )

    values[
        np.isfinite(values) & (values <= 0)
    ] = np.nan

    rows, columns = np.where(~np.isfinite(values))
    values[rows, columns] = medians[columns]

    values = np.clip(values, 1e-12, None)

    features = np.hstack([
        np.log10(values),
        censor_mask.astype(np.float32),
        missing_mask.astype(np.float32),
    ])

    return features, missing_elements


def predict_classes(
    dataframe: pd.DataFrame,
    model: XGBClassifier,
    preprocessing: dict,
    label_encoder,
    top_k: int = 3,
) -> pd.DataFrame:
    features, missing_elements = create_features(
        dataframe,
        preprocessing,
    )

    probabilities = model.predict_proba(features)
    number_of_classes = probabilities.shape[1]
    top_k = min(top_k, number_of_classes)

    rows = []

    for sample_index, sample_probabilities in enumerate(
        probabilities
    ):
        best_indices = np.argsort(
            sample_probabilities
        )[::-1][:top_k]

        result = {
            "sample_index": sample_index,
            "predicted_class": label_encoder.inverse_transform(
                [best_indices[0]]
            )[0],
            "confidence": float(
                sample_probabilities[best_indices[0]]
            ),
        }

        for rank, class_index in enumerate(
            best_indices,
            start=1,
        ):
            class_name = label_encoder.inverse_transform(
                [class_index]
            )[0]

            result[f"top_{rank}_class"] = class_name
            result[f"top_{rank}_probability"] = float(
                sample_probabilities[class_index]
            )

        rows.append(result)

    if missing_elements:
        print(
            "\nWarning: the following expected elements "
            "were absent and were imputed:"
        )
        print(", ".join(missing_elements))

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run inference using the trained "
            "OSNACA XGBoost classifier."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to an input CSV or Excel file.",
    )

    parser.add_argument(
        "--output",
        default=(
            "ml_experiment/"
            "prediction_results.csv"
        ),
        help="Location for the prediction results.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    model, preprocessing, label_encoder = (
        load_artifacts()
    )

    dataframe = load_uploaded_data(input_path)

    results = predict_classes(
        dataframe=dataframe,
        model=model,
        preprocessing=preprocessing,
        label_encoder=label_encoder,
        top_k=3,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(output_path, index=False)

    print("\nPredictions:")
    print(results.to_string(index=False))

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()