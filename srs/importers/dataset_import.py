"""Import uploaded CSV datasets into samples and element measurements."""

import csv
from io import StringIO
from math import isfinite

from django.db import close_old_connections, transaction
from django.utils import timezone

from srs.models import Dataset, Element, Sample, SampleMeasurement

# Accepted headings for the required sample fields.
SAMPLE_ID_COLUMNS = {"sample_id", "sample", "sample_code", "id"}
LATITUDE_COLUMNS = {"latitude", "lat", "y"}
LONGITUDE_COLUMNS = {"longitude", "long", "lon", "lng", "x"}
MEASUREMENT_BATCH_SIZE = 1000


def _normalise_lookup_name(name):
    """Normalise headings used to find sample and coordinate columns."""
    return str(name).lstrip("\ufeff").strip().lower().replace(" ", "_")


def _parse_float(value, label="Value"):
    """Parse a finite number, returning None for blanks and rejecting invalid text."""
    if value is None:
        return None

    value = str(value).strip()

    if value == "":
        return None

    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{label} must be numeric or blank. Invalid value: {value!r}."
        ) from exc

    if not isfinite(parsed):
        raise ValueError(
            f"{label} must be a finite numeric value."
        )

    return parsed


def _parse_coordinate(value, label, minimum, maximum):
    """Parse an optional latitude or longitude and check its range."""
    if value is None or str(value).strip() == "":
        return None

    parsed = _parse_float(value, label)

    if not minimum <= parsed <= maximum:
        raise ValueError(
            f"{label} must be between {minimum} and {maximum}."
        )

    return parsed


def _split_element_and_unit(column_name):
    """Split headings such as ``Cu_ppm`` into their element and unit."""
    column_name = str(column_name).strip()

    if "_" in column_name:
        parts = column_name.rsplit("_", 1)
        symbol = parts[0]
        unit = parts[1]
        return symbol, unit

    return column_name, ""


def _read_csv(dataset):
    """Return the headings and rows from a dataset's stored CSV file."""
    if not dataset.uploaded_file:
        raise ValueError("No uploaded file found for this dataset.")

    if not dataset.uploaded_file.name.lower().endswith(".csv"):
        raise ValueError(
            "Only CSV dataset uploads are currently supported."
        )

    with dataset.uploaded_file.open("rb") as file:
        try:
            contents = file.read().decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError(
                "CSV files must use UTF-8 text encoding."
            ) from exc

    try:
        reader = csv.DictReader(StringIO(contents))
        fieldnames = reader.fieldnames or []

        if not fieldnames:
            raise ValueError(
                "CSV file has no header row."
            )

        rows = list(reader)

    except csv.Error as exc:
        raise ValueError(
            "The CSV file could not be read. Please check its formatting."
        ) from exc

    if not rows:
        raise ValueError(
            "CSV must contain at least one sample row."
        )

    for row_number, row in enumerate(rows, start=2):
        if None in row:
            raise ValueError(
                f"Row {row_number} contains more values than the CSV header."
            )

    return fieldnames, rows


def _find_columns(fieldnames):
    """Find sample fields and return every remaining measurement heading."""
    if len(set(fieldnames)) != len(fieldnames):
        raise ValueError(
            "CSV headings must be unique."
        )

    normalised_fields = {}

    for field in fieldnames:
        key = _normalise_lookup_name(field)

        if key in normalised_fields:
            raise ValueError(
                f"CSV headings become duplicates after normalising: {field!r}."
            )

        normalised_fields[key] = field

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
            "CSV must contain a sample ID column, such as sample_id, "
            "sample, or sample_code."
        )

    ignored = {
        sample_id_col,
        latitude_col,
        longitude_col,
    }

    measurement_columns = [
        field
        for field in fieldnames
        if field not in ignored
    ]

    if not measurement_columns:
        raise ValueError(
            "CSV must contain at least one geochemical measurement column."
        )

    return (
        sample_id_col,
        latitude_col,
        longitude_col,
        measurement_columns,
    )


def _validate_sample_codes(rows, sample_id_col):
    """Reject missing or repeated sample IDs before changing stored rows."""
    seen = set()
    duplicates = set()

    for row in rows:
        sample_code = str(
            row.get(sample_id_col, "")
        ).strip()

        if not sample_code:
            raise ValueError(
                "One or more rows are missing a sample ID."
            )

        if sample_code in seen:
            duplicates.add(sample_code)

        seen.add(sample_code)

    if duplicates:
        duplicate_list = ", ".join(
            sorted(duplicates)[:10]
        )

        raise ValueError(
            f"Duplicate sample IDs found within this file: {duplicate_list}"
        )


def _resolve_measurement_columns(measurement_columns):
    """Resolve one Element row for each measurement heading."""
    column_details = []
    element_cache = {}

    for column in measurement_columns:
        symbol, unit = _split_element_and_unit(column)

        if not symbol:
            raise ValueError(
                f"Measurement heading has no element: {column!r}."
            )

        if symbol in element_cache:
            raise ValueError(
                f"Only one measurement column is allowed for element {symbol}."
            )

        element = element_cache.get(symbol)

        if element is None:
            element, _ = Element.objects.get_or_create(
                symbol=symbol,
                defaults={
                    "name": symbol,
                    "default_unit": unit,
                },
            )

            element_cache[symbol] = element

        column_details.append(
            (column, element, unit)
        )

    return column_details


def _flush_measurements(pending_measurements):
    """Save one batch of measurements and clear the reused list."""
    if not pending_measurements:
        return

    SampleMeasurement.objects.bulk_create(
        pending_measurements,
        batch_size=MEASUREMENT_BATCH_SIZE,
    )

    pending_measurements.clear()


def _replace_samples(
    dataset,
    rows,
    sample_id_col,
    latitude_col,
    longitude_col,
    column_details,
):
    """Replace saved samples and return the number of missing measurements."""
    Sample.objects.filter(
        dataset=dataset
    ).delete()

    pending_measurements = []
    null_count = 0

    for row_number, row in enumerate(rows, start=2):

        sample_code = str(
            row.get(sample_id_col, "")
        ).strip()

        sample = Sample.objects.create(
            dataset=dataset,
            sample_code=sample_code,
            latitude=(
                _parse_coordinate(
                    row.get(latitude_col),
                    "Latitude",
                    -90,
                    90,
                )
                if latitude_col
                else None
            ),
            longitude=(
                _parse_coordinate(
                    row.get(longitude_col),
                    "Longitude",
                    -180,
                    180,
                )
                if longitude_col
                else None
            ),
            metadata={
                "source_row": row
            },
        )

        for column, element, unit in column_details:

            value = _parse_float(
                row.get(column),
                label=(
                    f"{column} for sample "
                    f"{sample_code} on row {row_number}"
                ),
            )

            if value is None:
                null_count += 1

            pending_measurements.append(
                SampleMeasurement(
                    sample=sample,
                    element=element,
                    value=value,
                    unit=unit,
                )
            )

            if (
                len(pending_measurements)
                >= MEASUREMENT_BATCH_SIZE
            ):
                _flush_measurements(
                    pending_measurements
                )

    _flush_measurements(
        pending_measurements
    )

    return null_count


def _mark_completed(
    dataset,
    fieldnames,
    rows,
    sample_id_col,
    latitude_col,
    longitude_col,
    measurement_columns,
    null_count,
):
    """Store the final import counts and selected headings."""
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
        "measurements_imported": (
            len(rows)
            * len(measurement_columns)
        ),
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


def run_dataset_import(dataset_id):
    """Replace a dataset's samples with rows from its stored CSV file."""
    close_old_connections()

    try:
        dataset = Dataset.objects.get(
            id=dataset_id
        )

        dataset.status = Dataset.STATUS_RUNNING
        dataset.errors = []
        dataset.stats = {}
        dataset.completed_at = None

        dataset.save(
            update_fields=[
                "status",
                "errors",
                "stats",
                "completed_at",
            ]
        )

        try:
            fieldnames, rows = _read_csv(
                dataset
            )

            columns = _find_columns(
                fieldnames
            )

            (
                sample_id_col,
                latitude_col,
                longitude_col,
                measurement_columns,
            ) = columns

            _validate_sample_codes(
                rows,
                sample_id_col,
            )

            with transaction.atomic():

                column_details = (
                    _resolve_measurement_columns(
                        measurement_columns
                    )
                )

                null_count = _replace_samples(
                    dataset,
                    rows,
                    sample_id_col,
                    latitude_col,
                    longitude_col,
                    column_details,
                )

                _mark_completed(
                    dataset,
                    fieldnames,
                    rows,
                    sample_id_col,
                    latitude_col,
                    longitude_col,
                    measurement_columns,
                    null_count,
                )

        except Exception as exc:

            dataset.status = (
                Dataset.STATUS_FAILED
            )

            dataset.errors = [
                str(exc)
            ]

            dataset.completed_at = (
                timezone.now()
            )

            dataset.save(
                update_fields=[
                    "status",
                    "errors",
                    "completed_at",
                ]
            )

            raise

    finally:
        close_old_connections()