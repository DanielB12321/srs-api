"""
OSNACA workbook → SQL conversion pipeline.

Entry point: run_import(import_id)

Order of operations (matches the plan):
  1. Mark the ReferenceImport row as running
  2. Open both workbooks from the stored FileFields
  3. Seed Element / Mineral / DepositClassification if missing
  4. Upsert ReferenceDeposit rows from Deposit Locations
  5. Upsert ReferenceSample rows from Ore samples + CSIRO Cloncurry Samples
  6. Bulk-insert ReferenceSampleMeasurement rows from Data + Cloncurry Supplement + PGEs
  7. Record stats / errors, mark completed
"""
from __future__ import annotations

import re

import openpyxl
from django.db import transaction
from django.utils import timezone


_NUMERIC_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _to_float(value) -> float | None:
    """
    Coerce a spreadsheet cell to a float, tolerating dirty data.
    Strips trailing junk like degree signs, asterisks, or stray whitespace.
    Returns None for blanks or unparseable values.
    """
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = _NUMERIC_RE.search(str(value))
    return float(match.group(0)) if match else None

from ..models import (
    DepositClassification,
    Element,
    Mineral,
    ReferenceDeposit,
    ReferenceImport,
    ReferenceSample,
    ReferenceSampleMeasurement,
)
from .headers import flatten_headers, header_label
from .seed_data import ELEMENTS, ELEMENT_METHOD_OVERRIDES, NON_ELEMENT_COLUMNS


# top level entry 

def run_import(import_id: int) -> ReferenceImport:
    # Ensure this thread has a fresh DB connection (required when called from a background thread)
    from django.db import close_old_connections
    close_old_connections()

    import_row = ReferenceImport.objects.get(pk=import_id)
    import_row.status = ReferenceImport.STATUS_RUNNING
    import_row.errors = []
    import_row.save(update_fields=["status", "errors"])

    stats = {"deposits": 0, "samples": 0, "measurements": 0, "warnings": 0}
    errors: list[dict] = []

    try:
        with transaction.atomic():
            data_wb = openpyxl.load_workbook(
                import_row.data_file.path, read_only=True, data_only=True
            )
            meta_wb = openpyxl.load_workbook(
                import_row.metadata_file.path, read_only=True, data_only=True
            )

            seed_lookups(meta_wb)
            deposits_by_code = load_deposits(meta_wb, import_row, stats)
            samples_by_code = load_samples(meta_wb, import_row, deposits_by_code, stats, errors)
            load_measurements(data_wb, import_row, samples_by_code, stats, errors)

        import_row.status = ReferenceImport.STATUS_COMPLETED
    except Exception as exc:                                   
        import_row.status = ReferenceImport.STATUS_FAILED
        errors.append({"phase": "fatal", "reason": str(exc)})

    stats["warnings"] = len(errors)
    import_row.stats = stats
    import_row.errors = errors
    import_row.completed_at = timezone.now()
    import_row.save(update_fields=["status", "stats", "errors", "completed_at"])
    return import_row


# seed lookups

def seed_lookups(meta_wb) -> None:
    """Idempotent. Run on every import; no-op if rows already exist."""
    for symbol, name, atomic_number, default_unit in ELEMENTS:
        Element.objects.update_or_create(
            symbol=symbol,
            defaults={
                "name": name,
                "atomic_number": atomic_number,
                "default_unit": default_unit,
            },
        )

    ws = meta_wb["Mineral Codes"]
    for name, code, *_ in ws.iter_rows(min_row=2, values_only=True):
        if not code or not name:
            continue
        Mineral.objects.update_or_create(code=code, defaults={"name": name})

    ws = meta_wb["Ore Deposit Classification"]
    current_class = None
    for cls, sub, notes in ws.iter_rows(min_row=2, values_only=True):
        if cls:
            current_class = cls
        if current_class is None:
            continue
        DepositClassification.objects.update_or_create(
            deposit_class=current_class,
            sub_class=sub or "",
            defaults={"notes": notes or ""},
        )


# deposits

def load_deposits(meta_wb, import_row, stats) -> dict[str, ReferenceDeposit]:
    ws = meta_wb["Deposit Locations"]
    rows = list(ws.iter_rows(min_row=2, values_only=True))
    deposits_by_code: dict[str, ReferenceDeposit] = {}

    for name, three_char, cls, sub, lng, lat in rows:
        if not name or not three_char:
            continue
        deposit, _ = ReferenceDeposit.objects.update_or_create(
            name=name,
            three_char_code=three_char,
            defaults={
                "import_ref": import_row,
                "deposit_type": cls or "",
                "mineral_system": sub or "",
                "latitude": _to_float(lat),
                "longitude": _to_float(lng),
                "source": "OSNACA",
            },
        )
        deposits_by_code[three_char] = deposit

    stats["deposits"] = len(deposits_by_code)
    return deposits_by_code


# samples 

def load_samples(meta_wb, import_row, deposits_by_code, stats, errors):
    samples_by_code: dict[str, ReferenceSample] = {}
    for sheet_name in ("Ore samples", "CSIRO Cloncurry Samples"):
        ws = meta_wb[sheet_name]
        # row 1 = headers, row 2 = sub-headers, data starts row 3
        for row_idx, row in enumerate(ws.iter_rows(min_row=3, values_only=True), start=3):
            sample_code = row[0]
            three_char = row[4]
            if sample_code in (None, ""):
                continue

            deposit = deposits_by_code.get(three_char)
            if deposit is None and three_char:
                errors.append({
                    "sheet": sheet_name, "row": row_idx,
                    "reason": f"Unknown three_char_code: {three_char!r}",
                })

            sample, _ = ReferenceSample.objects.update_or_create(
                import_ref=import_row,
                sample_code=str(sample_code),
                defaults={
                    "reference_deposit": deposit,
                    "sample_type": f"{row[7] or ''} / {row[8] or ''}".strip(" /"),
                    "latitude": _to_float(row[11]),       # col L
                    "longitude": _to_float(row[10]),      # col K
                    "source_dataset": "OSNACA",
                    "source_reference": row[57] or "",    # col 58
                    "metadata": _build_sample_metadata(row),
                },
            )
            samples_by_code[str(sample_code)] = sample

    stats["samples"] = len(samples_by_code)
    return samples_by_code


def _build_sample_metadata(row) -> dict:
    """Pack the long-tail columns into the JSON blob.  See plan section 6."""
    mineral_codes = [c for c in row[17:25] if c]   # MIN 1-8 (cols R-Y)
    minerals = list(
        Mineral.objects.filter(code__in=mineral_codes).values_list("name", flat=True)
    )

    texture_headers = [
        "Massive-Homogenous", "Massive-Banded", "Massive-Granular",
        "Matrix or semi-massive", "Disseminated", "Blebs-Patches",
        "Stringer-Sulphide Veinlets", "Open space sulphide", "Breccia",
        "Sheared", "Vein Stockwork", "Vein(s) Crack-Seal", "Vein(s) Cockscomb",
        "Vein(s) Massive", "Vein(s) Shear Banded", "Vein(s) Breccia",
    ]
    textures = [
        label for label, cell in zip(texture_headers, row[25:41]) if cell == 1
    ]

    resource_headers = [
        "Mt", "Au_g_t", "Ag_g_t", "Pt_g_t", "Pd_g_t",
        "Cu_pct", "Zn_pct", "Pb_pct", "Ni_pct", "Co_pct",
        "Mo_pct", "W_pct", "Sn_pct", "U_pct", "Fe_pct", "Mn_pct",
    ]
    global_resource = {k: v for k, v in zip(resource_headers, row[41:57])}

    return {
        "donor": row[3],
        "your_sample_id": row[9],
        "utm_e": row[12], "utm_n": row[13],
        "rl": row[14], "srtm_rl": row[15],
        "description": row[16],
        "ore_minerals": minerals,
        "ore_textures": textures,
        "global_resource": global_resource,
    }


# measurements 

def load_measurements(data_wb, import_row, samples_by_code, stats, errors):
    elements_by_symbol = {e.symbol: e for e in Element.objects.all()}
    rows_to_create: list[ReferenceSampleMeasurement] = []

    for sheet_name in ("Data", "Cloncurry Supplement", "PGEs"):
        if sheet_name not in data_wb.sheetnames:
            continue
        ws = data_wb[sheet_name]
        all_rows = list(ws.iter_rows(values_only=True))
        if len(all_rows) < 3:
            continue

        # Headers: row 1 = group, row 2 = detail. Data from row 3.
        if sheet_name == "PGEs":
            # PGE sheet has a single header row
            headers = [("PGE", h or "") for h in all_rows[0]]
            data_start = 1
        else:
            headers = flatten_headers(all_rows[0], all_rows[1])
            data_start = 2

        for row in all_rows[data_start:]:
            sample_code = row[0]
            if sample_code in (None, ""):
                continue
            sample = samples_by_code.get(str(sample_code))
            if sample is None:
                errors.append({
                    "sheet": sheet_name,
                    "sample_code": sample_code,
                    "reason": "No matching ReferenceSample (missing in metadata workbook)",
                })
                continue

            for col_idx, (group, detail) in enumerate(headers):
                label = header_label(group, detail)
                if label in NON_ELEMENT_COLUMNS or label == "":
                    continue

                symbol, method = ELEMENT_METHOD_OVERRIDES.get(
                    (group, detail), (label, "")
                )
                element = elements_by_symbol.get(symbol)
                if element is None:
                    continue   # not an element column (e.g. Deposit/Class on PGE sheet)

                cell = row[col_idx]
                if cell in (None, ""):
                    continue
                if not isinstance(cell, (int, float)):
                    # Non-numeric marker like "IS" (Insufficient Sample). Skip with warning.
                    errors.append({
                        "sheet": sheet_name,
                        "sample_code": sample_code,
                        "column": label,
                        "reason": f"Non-numeric measurement value: {cell!r}",
                    })
                    continue

                below = False
                detection_limit = None
                value = cell
                if cell < 0:
                    below = True
                    detection_limit = abs(cell)
                    value = None

                rows_to_create.append(ReferenceSampleMeasurement(
                    import_ref=import_row,
                    reference_sample=sample,
                    element=element,
                    analytical_method=method,
                    value=value,
                    unit=element.default_unit,
                    below_detection_limit=below,
                    detection_limit=detection_limit,
                ))

    # Bulk insert for speed — 120k rows in one round trip
    ReferenceSampleMeasurement.objects.bulk_create(
        rows_to_create,
        batch_size=2000,
        ignore_conflicts=True,        # tolerates re-runs against same import
    )
    stats["measurements"] = len(rows_to_create)
