"""Analysis-wide relationships between uploaded sample elements."""

from math import sqrt

from .preprocessing import extract_values, prepare_vectors, resolve_options


MIN_SHARED_SAMPLES = 3
STRONG_CORRELATION = 0.70
MAX_MATRIX_ELEMENTS = 40


def _pearson(left, right):
    """Return Pearson's r, or None when either side has no variation."""
    if len(left) < MIN_SHARED_SAMPLES or len(left) != len(right):
        return None

    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_centred = [value - left_mean for value in left]
    right_centred = [value - right_mean for value in right]
    numerator = sum(
        left_value * right_value
        for left_value, right_value in zip(left_centred, right_centred)
    )
    left_spread = sum(value * value for value in left_centred)
    right_spread = sum(value * value for value in right_centred)
    denominator = sqrt(left_spread * right_spread)

    if denominator == 0:
        return None

    # Floating-point rounding can place an otherwise valid result just outside
    # the mathematical -1 to 1 range.
    return max(-1.0, min(1.0, numerator / denominator))


def _transformed_sample_values(sample, options):
    """Use the same cleaning and transformation choices as the ranking run."""
    values, _ = extract_values(sample.get("measurements") or [], options)
    if not values:
        return {}

    prepared = prepare_vectors(values, values, set(values), options)
    return dict(zip(prepared.symbols, prepared.input_vector))


def _value_space(options):
    if options.get("normalise"):
        return "clr"
    if options.get("log_transform"):
        return "log10"
    return "raw"


def calculate_element_associations(
    samples,
    preprocessing=None,
    selected_elements=None,
):
    """Calculate pairwise element correlations across uploaded samples.

    Each relationship uses only samples containing valid values for both
    elements. At least three shared samples are required, and the sample count
    is returned beside every result so small groups are easy to recognise.
    """
    options = resolve_options(preprocessing, selected_elements)
    transformed_samples = [
        _transformed_sample_values(sample, options)
        for sample in samples
    ]
    coverage = {}
    for values in transformed_samples:
        for symbol in values:
            coverage[symbol] = coverage.get(symbol, 0) + 1

    eligible_elements = sorted(
        symbol
        for symbol, count in coverage.items()
        if count >= MIN_SHARED_SAMPLES
    )
    correlations = {}
    shared_counts = {}
    associations = []

    for left_index, left_symbol in enumerate(eligible_elements):
        for right_symbol in eligible_elements[left_index + 1:]:
            pairs = [
                (values[left_symbol], values[right_symbol])
                for values in transformed_samples
                if left_symbol in values and right_symbol in values
            ]
            shared_count = len(pairs)
            if shared_count < MIN_SHARED_SAMPLES:
                continue

            correlation = _pearson(
                [pair[0] for pair in pairs],
                [pair[1] for pair in pairs],
            )
            if correlation is None:
                continue

            correlation = round(correlation, 6)
            key = (left_symbol, right_symbol)
            correlations[key] = correlation
            shared_counts[key] = shared_count
            absolute_correlation = abs(correlation)
            associations.append({
                "element_a": left_symbol,
                "element_b": right_symbol,
                "correlation": correlation,
                "absolute_correlation": round(absolute_correlation, 6),
                "direction": "positive" if correlation >= 0 else "negative",
                "strength": (
                    "strong"
                    if absolute_correlation >= STRONG_CORRELATION
                    else "moderate"
                    if absolute_correlation >= 0.40
                    else "weak"
                ),
                "shared_sample_count": shared_count,
            })

    associations.sort(key=lambda row: (
        -row["absolute_correlation"],
        row["element_a"],
        row["element_b"],
    ))

    strongest_by_element = {symbol: 0.0 for symbol in eligible_elements}
    for row in associations:
        strength = row["absolute_correlation"]
        strongest_by_element[row["element_a"]] = max(
            strongest_by_element[row["element_a"]], strength
        )
        strongest_by_element[row["element_b"]] = max(
            strongest_by_element[row["element_b"]], strength
        )

    # Very wide assay files make unreadable heatmaps. Keep the most informative
    # elements in the matrix while retaining every valid pair in associations.
    matrix_elements = sorted(
        sorted(
            eligible_elements,
            key=lambda symbol: (
                -strongest_by_element[symbol],
                -coverage[symbol],
                symbol,
            ),
        )[:MAX_MATRIX_ELEMENTS]
    )
    matrix = []
    count_matrix = []
    for left_symbol in matrix_elements:
        correlation_row = []
        count_row = []
        for right_symbol in matrix_elements:
            if left_symbol == right_symbol:
                own_values = [
                    values[left_symbol]
                    for values in transformed_samples
                    if left_symbol in values
                ]
                value = _pearson(own_values, own_values)
                correlation_row.append(1.0 if value is not None else None)
                count_row.append(len(own_values))
                continue

            key = tuple(sorted((left_symbol, right_symbol)))
            correlation_row.append(correlations.get(key))
            count_row.append(shared_counts.get(key, 0))
        matrix.append(correlation_row)
        count_matrix.append(count_row)

    strong_count = sum(
        row["strength"] == "strong"
        for row in associations
    )
    if len(samples) < MIN_SHARED_SAMPLES:
        message = (
            "At least three analysed samples are required to calculate "
            "element associations."
        )
    elif not associations:
        message = (
            "No element pairs had enough shared, varying measurements to "
            "calculate a correlation."
        )
    else:
        message = (
            f"Calculated {len(associations)} element relationships across "
            f"{len(samples)} analysed samples. {strong_count} had an absolute "
            f"correlation of {STRONG_CORRELATION:.2f} or higher."
        )

    return {
        "schema_version": "1.0",
        "method": "pearson",
        "value_space": _value_space(options),
        "sample_count": len(samples),
        "minimum_shared_samples": MIN_SHARED_SAMPLES,
        "strong_threshold": STRONG_CORRELATION,
        "available": bool(associations),
        "message": message,
        "eligible_elements": eligible_elements,
        "matrix_elements": matrix_elements,
        "matrix": matrix,
        "shared_sample_counts": count_matrix,
        "matrix_omitted_elements": [
            symbol for symbol in eligible_elements
            if symbol not in matrix_elements
        ],
        "strong_association_count": strong_count,
        "associations": associations,
    }
