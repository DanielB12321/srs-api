"""Build and store geochemical signatures using the analysis preprocessing."""

from statistics import median

from django.db import transaction

from .models import GeochemicalSignature
from .preprocessing import (
    PIPELINE_VERSION,
    describe,
    extract_values,
    prepare_vectors,
    resolve_options,
)


def _signature_values(measurements, options):
    """Keep the cleaned ppm values alongside the values used for comparison."""
    raw_values, imputed = extract_values(measurements, options)
    if not raw_values:
        return {}, {}, []

    # Use the same sample on both sides to apply the normal analysis transforms.
    prepared = prepare_vectors(
        raw_values,
        raw_values,
        set(raw_values),
        options,
        imputed,
    )
    transformed = dict(zip(prepared.symbols, prepared.input_vector))
    ordered_raw = {symbol: raw_values[symbol] for symbol in prepared.symbols}
    return ordered_raw, transformed, sorted(imputed)


def _value_space(options):
    """Name the scale used by the stored comparison vector."""
    if options.get("normalise"):
        return "clr"
    if options.get("log_transform"):
        return "log10_ppm"
    return "ppm"


def build_analysis_signatures(samples, preprocessing=None, selected_elements=None):
    """Build one profile per sample and a median profile for the full run."""
    options = resolve_options(preprocessing, selected_elements)
    profiles = []
    sample_raw_values = []

    for sample_index, sample in enumerate(samples):
        raw_vector, vector, imputed = _signature_values(
            sample.get("measurements") or [],
            options,
        )
        if not vector:
            continue

        sample_raw_values.append(raw_vector)
        profiles.append({
            "kind": GeochemicalSignature.KIND_SAMPLE,
            "sample_index": sample_index,
            "sample_code": str(
                sample.get("sample_code") or sample.get("name") or ""
            ),
            "representative_method": "individual_sample",
            "value_space": _value_space(options),
            "elements": list(vector),
            "raw_vector": raw_vector,
            "vector": vector,
            "imputed_elements": imputed,
            "preprocessing": describe(options, vector),
            "pipeline_version": PIPELINE_VERSION,
        })

    all_elements = sorted({
        symbol
        for values in sample_raw_values
        for symbol in values
    })
    # Take each element's median in ppm first, then transform that combined profile.
    # Missing values are left out, rather than counted as zero.
    composite_raw = {
        symbol: median([
            values[symbol]
            for values in sample_raw_values
            if symbol in values
        ])
        for symbol in all_elements
    }
    if composite_raw:
        prepared = prepare_vectors(
            composite_raw,
            composite_raw,
            set(composite_raw),
            options,
        )
        composite_vector = dict(zip(prepared.symbols, prepared.input_vector))
        profiles.append({
            "kind": GeochemicalSignature.KIND_COMPOSITE,
            "sample_index": -1,
            "sample_code": "",
            "representative_method": "median",
            "value_space": _value_space(options),
            "elements": prepared.symbols,
            "raw_vector": {
                symbol: composite_raw[symbol]
                for symbol in prepared.symbols
            },
            "vector": composite_vector,
            "imputed_elements": [],
            "preprocessing": describe(options, prepared.symbols),
            "pipeline_version": PIPELINE_VERSION,
        })

    return profiles


@transaction.atomic
def save_analysis_signatures(
    full_analysis,
    samples,
    preprocessing=None,
    selected_elements=None,
):
    """Replace the profiles for a run so retrying cannot create duplicates."""
    profiles = build_analysis_signatures(
        samples,
        preprocessing,
        selected_elements,
    )
    # The transaction keeps the old profiles if saving their replacements fails.
    full_analysis.signatures.all().delete()
    GeochemicalSignature.objects.bulk_create([
        GeochemicalSignature(full_analysis=full_analysis, **profile)
        for profile in profiles
    ])
    return profiles


def serialize_signature(signature):
    """Return a stable API representation of a stored signature."""
    return {
        "id": signature.id,
        "kind": signature.kind,
        "sample_index": (
            signature.sample_index
            if signature.kind == GeochemicalSignature.KIND_SAMPLE
            else None
        ),
        "sample_code": signature.sample_code,
        "representative_method": signature.representative_method,
        "value_space": signature.value_space,
        "elements": signature.elements,
        "raw_vector": signature.raw_vector,
        "vector": signature.vector,
        "imputed_elements": signature.imputed_elements,
        "preprocessing": signature.preprocessing,
        "pipeline_version": signature.pipeline_version,
        "created_at": signature.created_at,
    }
