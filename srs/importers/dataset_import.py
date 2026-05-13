import csv
import os

from django.db import transaction
from django.utils import timezone

from srs.models import Dataset, Sample, SampleMeasurement, Element

# Development version of dataset import functionality
# This is currently only designed to handle CSV files, and expects a fairly specific format:
SAMPLE_ID_COLUMNS = {"sample_id", "sample", "sample_code", "id"}
LATITUDE_COLUMNS = {"latitude", "lat", "y"}
LONGITUDE_COLUMNS = {"longitude", "long", "lon", "lng", "x"}


def _normalise_column_name(name):
    return str(name).strip()


def _normalise_lookup_name(name):
    return str(name).strip().lower().replace(" ", "_")


def _parse_float(value):
    if value is None:
        return None

    value = str(value).strip()

    if value == "":
        return None

    try:
        return float(value)
    except ValueError:
        return None


def _split_element_and_unit(column_name):
    column_name = str(column_name).strip()

    if "_" in column_name:
        parts = column_name.rsplit("_", 1)
        symbol = parts[0]
        unit = parts[1]
        return symbol, unit

    return column_name, ""


def run_dataset_import(dataset_id):
    dataset = Dataset.objects.get(id=dataset_id)

    dataset.status = Dataset.STATUS_RUNNING
    dataset.errors = []
    dataset.stats = {}
    dataset.completed_at = None
    dataset.save(update_fields=["status", "errors", "stats", "completed_at"])

    try:
        if not dataset.uploaded_file:
            raise ValueError("No uploaded file found for this dataset.")

        file_name = dataset.uploaded_file.name.lower()

        if not file_name.endswith(".csv"):
            raise ValueError("Only CSV dataset uploads are currently supported.")

        with dataset.uploaded_file.open("r") as file:
            reader = csv.DictReader(file)
            fieldnames = reader.fieldnames or []

            if not fieldnames:
                raise ValueError("CSV file has no header row.")

            rows = list(reader)

        normalised_fields = {
            _normalise_lookup_name(field): field
            for field in fieldnames
        }

        sample_id_col = None
        latitude_col = None
        longitude_col = None

        for key, original in normalised_fields.items():
            if key in SAMPLE_ID_COLUMNS:
                sample_id_col = original
            elif key in LATITUDE_COLUMNS:
                latitude_col = original
            elif key in LONGITUDE_COLUMNS:
                longitude_col = original

        if not sample_id_col:
            raise ValueError(
                "CSV must contain a sample ID column, such as sample_id, sample, or sample_code."
            )

        measurement_columns = []
        ignored_columns = {sample_id_col}

        if latitude_col:
            ignored_columns.add(latitude_col)

        if longitude_col:
            ignored_columns.add(longitude_col)

        for field in fieldnames:
            if field not in ignored_columns:
                measurement_columns.append(field)

        if not measurement_columns:
            raise ValueError("CSV must contain at least one geochemical measurement column.")

        seen_sample_codes = set()
        duplicate_sample_codes = []

        for row in rows:
            sample_code = str(row.get(sample_id_col, "")).strip()

            if not sample_code:
                raise ValueError("One or more rows are missing a sample ID.")

            if sample_code in seen_sample_codes:
                duplicate_sample_codes.append(sample_code)

            seen_sample_codes.add(sample_code)

        if duplicate_sample_codes:
            duplicate_list = ", ".join(sorted(set(duplicate_sample_codes))[:10])
            raise ValueError(
                f"Duplicate sample IDs found within this file: {duplicate_list}"
            )

        null_count = 0

        with transaction.atomic():
            Sample.objects.filter(dataset=dataset).delete()

            for row in rows:
                sample_code = str(row.get(sample_id_col, "")).strip()

                sample = Sample.objects.create(
                    dataset=dataset,
                    sample_code=sample_code,
                    latitude=_parse_float(row.get(latitude_col)) if latitude_col else None,
                    longitude=_parse_float(row.get(longitude_col)) if longitude_col else None,
                    metadata={
                        "source_row": row,
                    },
                )

                for column in measurement_columns:
                    raw_value = row.get(column)

                    if raw_value is None or str(raw_value).strip() == "":
                        null_count += 1
                        value = None
                    else:
                        value = _parse_float(raw_value)

                        if value is None:
                            null_count += 1

                    symbol, unit = _split_element_and_unit(column)

                    element, _ = Element.objects.get_or_create(
                        symbol=symbol,
                        defaults={
                            "name": symbol,
                            "default_unit": unit,
                        },
                    )

                    SampleMeasurement.objects.create(
                        sample=sample,
                        element=element,
                        value=value,
                        unit=unit,
                    )

        dataset.row_count = len(rows)
        dataset.col_count = len(fieldnames)
        dataset.null_count = null_count
        dataset.status = Dataset.STATUS_COMPLETED
        dataset.completed_at = timezone.now()
        dataset.errors = []
        dataset.stats = {
            "sample_id_column": sample_id_col,
            "latitude_column": latitude_col,
            "longitude_column": longitude_col,
            "measurement_columns": measurement_columns,
            "rows_imported": len(rows),
            "measurements_imported": len(rows) * len(measurement_columns),
        }
        dataset.save(
            update_fields=[
                "row_count",
                "col_count",
                "null_count",
                "status",
                "completed_at",
                "errors",
                "stats",
            ]
        )

    except Exception as exc:
        dataset.status = Dataset.STATUS_FAILED
        dataset.errors = [str(exc)]
        dataset.completed_at = timezone.now()
        dataset.save(update_fields=["status", "errors", "completed_at"])
        raise