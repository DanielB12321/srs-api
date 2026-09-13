"""Work out which elements rise or fall together across the uploaded samples."""

from math import exp, isfinite, lgamma, log, log1p, sqrt

from .preprocessing import extract_values, prepare_vectors, resolve_options


MINIMUM_POSSIBLE_SAMPLES = 3
DEFAULT_MINIMUM_SAMPLES = 3
DEFAULT_STRONG_CORRELATION = 0.70
DEFAULT_MAXIMUM_ADJUSTED_P = 0.05
MAX_MATRIX_ELEMENTS = 40
SUPPORTED_METHODS = {"pearson", "spearman"}


def resolve_association_options(raw_options=None):
    """Validate saved/requested association settings and fill their defaults."""
    if raw_options in (None, ""):
        raw_options = {}
    if not isinstance(raw_options, dict):
        raise ValueError("The element-association options must be an object.")

    method = str(raw_options.get("method", "pearson")).strip().lower()
    if method not in SUPPORTED_METHODS:
        raise ValueError("The association method must be Pearson or Spearman.")

    try:
        minimum_samples = int(
            raw_options.get("minimum_shared_samples", DEFAULT_MINIMUM_SAMPLES)
        )
        minimum_correlation = float(
            raw_options.get(
                "minimum_absolute_correlation",
                DEFAULT_STRONG_CORRELATION,
            )
        )
        maximum_p = float(
            raw_options.get(
                "maximum_adjusted_p_value",
                DEFAULT_MAXIMUM_ADJUSTED_P,
            )
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Association sample, correlation and p-value limits must be numbers."
        ) from error

    if minimum_samples < MINIMUM_POSSIBLE_SAMPLES:
        raise ValueError("Element associations require at least three shared samples.")
    if not isfinite(minimum_correlation) or not 0 <= minimum_correlation <= 1:
        raise ValueError("Minimum absolute correlation must be between 0 and 1.")
    if not isfinite(maximum_p) or not 0 < maximum_p <= 1:
        raise ValueError("Maximum adjusted p-value must be greater than 0 and at most 1.")

    return {
        "method": method,
        "minimum_shared_samples": minimum_samples,
        "minimum_absolute_correlation": minimum_correlation,
        "maximum_adjusted_p_value": maximum_p,
    }


def _pearson(left, right):
    """Return Pearson's r, or None when either side has no variation."""
    if len(left) < MINIMUM_POSSIBLE_SAMPLES or len(left) != len(right):
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

    return max(-1.0, min(1.0, numerator / denominator))


def _average_ranks(values):
    """Return one-based ranks, averaging positions occupied by tied values."""
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    position = 0

    while position < len(ordered):
        end = position + 1
        while end < len(ordered) and ordered[end][1] == ordered[position][1]:
            end += 1
        average_rank = ((position + 1) + end) / 2
        for original_index, _ in ordered[position:end]:
            ranks[original_index] = average_rank
        position = end

    return ranks


def _correlation(left, right, method):
    # Spearman uses the order of the values instead of their actual concentrations.
    if method == "spearman":
        return _pearson(_average_ranks(left), _average_ranks(right))
    return _pearson(left, right)


def _beta_continued_fraction(a, b, x):
    """Evaluate the continued fraction used by the incomplete beta function."""
    # This is the numerical part of the p-value calculation below.
    # Stop once the answer settles, and keep denominators away from zero.
    maximum_iterations = 200
    tolerance = 3e-14
    tiny = 1e-300
    qab = a + b
    qap = a + 1
    qam = a - 1
    c = 1.0
    d = 1.0 - (qab * x / qap)
    d = tiny if abs(d) < tiny else d
    d = 1.0 / d
    result = d

    for iteration in range(1, maximum_iterations + 1):
        doubled = 2 * iteration
        coefficient = (
            iteration * (b - iteration) * x
            / ((qam + doubled) * (a + doubled))
        )
        d = 1.0 + coefficient * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + coefficient / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        result *= d * c

        coefficient = -(
            (a + iteration) * (qab + iteration) * x
            / ((a + doubled) * (qap + doubled))
        )
        d = 1.0 + coefficient * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + coefficient / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        change = d * c
        result *= change
        if abs(change - 1.0) < tolerance:
            break

    return result


def _regularized_incomplete_beta(a, b, x):
    """Work out the beta function needed for the correlation p-value."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0

    front = exp(
        lgamma(a + b) - lgamma(a) - lgamma(b)
        + a * log(x) + b * log1p(-x)
    )
    if x < (a + 1) / (a + b + 2):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1 - front * _beta_continued_fraction(b, a, 1 - x) / b


def _two_sided_p_value(correlation, sample_count):
    """Calculate the usual two-sided t-test p-value for a correlation."""
    if sample_count < MINIMUM_POSSIBLE_SAMPLES:
        return None
    absolute = abs(correlation)
    if absolute >= 1:
        return 0.0
    degrees_of_freedom = sample_count - 2
    return max(0.0, min(1.0, _regularized_incomplete_beta(
        degrees_of_freedom / 2,
        0.5,
        1 - (absolute * absolute),
    )))


def _apply_false_discovery_rate(associations):
    """Add Benjamini-Hochberg adjusted p-values across every tested pair."""
    ordered = sorted(
        enumerate(associations),
        key=lambda item: item[1]["p_value"],
    )
    total = len(ordered)
    running_minimum = 1.0

    # Adjust for testing lots of pairs, working backwards to keep the limits consistent.
    for reverse_index in range(total - 1, -1, -1):
        original_index, row = ordered[reverse_index]
        rank = reverse_index + 1
        adjusted = min(1.0, row["p_value"] * total / rank)
        running_minimum = min(running_minimum, adjusted)
        associations[original_index]["adjusted_p_value"] = round(
            running_minimum,
            8,
        )


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


def _reliability(sample_count):
    if sample_count < 10:
        return "limited"
    if sample_count < 30:
        return "moderate"
    return "high"


def calculate_element_associations(
    samples,
    preprocessing=None,
    selected_elements=None,
    association_options=None,
):
    """Calculate pairwise element correlations across uploaded samples.

    Every pair carries its own shared-sample count and an FDR-adjusted p-value.
    This prevents missing measurements and large numbers of element pairs from
    making the confidence shown in the report look stronger than it is.
    """
    options = resolve_options(preprocessing, selected_elements)
    filters = resolve_association_options(association_options)
    method = filters["method"]
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
        if count >= MINIMUM_POSSIBLE_SAMPLES
    )
    correlations = {}
    shared_counts = {}
    associations = []

    for left_index, left_symbol in enumerate(eligible_elements):
        for right_symbol in eligible_elements[left_index + 1:]:
            # Only compare rows that have a measurement for both elements.
            pairs = [
                (values[left_symbol], values[right_symbol])
                for values in transformed_samples
                if left_symbol in values and right_symbol in values
            ]
            shared_count = len(pairs)
            if shared_count < MINIMUM_POSSIBLE_SAMPLES:
                continue

            correlation = _correlation(
                [pair[0] for pair in pairs],
                [pair[1] for pair in pairs],
                method,
            )
            if correlation is None:
                continue

            # Work out the p-value before rounding, so almost-perfect isn't treated as perfect.
            p_value = _two_sided_p_value(correlation, shared_count)
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
                "p_value": round(p_value, 8),
                "direction": "positive" if correlation >= 0 else "negative",
                "strength": (
                    "strong"
                    if absolute_correlation >= DEFAULT_STRONG_CORRELATION
                    else "moderate"
                    if absolute_correlation >= 0.40
                    else "weak"
                ),
                "shared_sample_count": shared_count,
                "reliability": _reliability(shared_count),
            })

    _apply_false_discovery_rate(associations)
    for row in associations:
        row["passes_filters"] = (
            row["shared_sample_count"] >= filters["minimum_shared_samples"]
            and row["absolute_correlation"]
            >= filters["minimum_absolute_correlation"]
            and row["adjusted_p_value"]
            <= filters["maximum_adjusted_p_value"]
        )
    associations.sort(key=lambda row: (
        not row["passes_filters"],
        -row["absolute_correlation"],
        row["adjusted_p_value"],
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

    # Keep the heatmap to 40 useful elements. The full pair list is still saved.
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
    filtered_matrix = []
    count_matrix = []
    passing_pairs = {
        tuple(sorted((row["element_a"], row["element_b"])))
        for row in associations
        if row["passes_filters"]
    }
    for left_symbol in matrix_elements:
        correlation_row = []
        filtered_row = []
        count_row = []
        for right_symbol in matrix_elements:
            if left_symbol == right_symbol:
                own_values = [
                    values[left_symbol]
                    for values in transformed_samples
                    if left_symbol in values
                ]
                value = _correlation(own_values, own_values, method)
                diagonal = 1.0 if value is not None else None
                correlation_row.append(diagonal)
                filtered_row.append(
                    diagonal
                    if len(own_values) >= filters["minimum_shared_samples"]
                    else None
                )
                count_row.append(len(own_values))
                continue

            key = tuple(sorted((left_symbol, right_symbol)))
            value = correlations.get(key)
            correlation_row.append(value)
            filtered_row.append(value if key in passing_pairs else None)
            count_row.append(shared_counts.get(key, 0))
        matrix.append(correlation_row)
        filtered_matrix.append(filtered_row)
        count_matrix.append(count_row)

    filtered_count = sum(row["passes_filters"] for row in associations)
    strong_count = sum(
        row["strength"] == "strong"
        for row in associations
    )
    warnings = []
    if len(samples) < 10:
        warnings.append(
            "Fewer than 10 samples were analysed, so correlations have limited reliability."
        )
    if method == "spearman":
        warnings.append(
            "Spearman p-values use the standard two-sided t approximation."
        )
    omitted_elements = [
        symbol for symbol in eligible_elements
        if symbol not in matrix_elements
    ]
    if omitted_elements:
        warnings.append(
            f"The heatmap shows the {MAX_MATRIX_ELEMENTS} most informative elements; "
            "all valid pairs remain available in the saved association list."
        )

    if len(samples) < MINIMUM_POSSIBLE_SAMPLES:
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
            f"Calculated {len(associations)} {method.title()} relationships "
            f"across {len(samples)} analysed samples. {filtered_count} met the "
            "selected strength, sample-count and adjusted p-value filters."
        )

    return {
        "schema_version": "1.1",
        "method": method,
        "p_value_method": "two_sided_student_t",
        "multiple_testing_correction": "benjamini_hochberg",
        "value_space": _value_space(options),
        "sample_count": len(samples),
        "minimum_possible_shared_samples": MINIMUM_POSSIBLE_SAMPLES,
        "filters": filters,
        "strong_threshold": DEFAULT_STRONG_CORRELATION,
        "available": bool(associations),
        "message": message,
        "warnings": warnings,
        "eligible_elements": eligible_elements,
        "matrix_elements": matrix_elements,
        "matrix": matrix,
        "filtered_matrix": filtered_matrix,
        "shared_sample_counts": count_matrix,
        "matrix_omitted_elements": omitted_elements,
        "strong_association_count": strong_count,
        "filtered_association_count": filtered_count,
        "associations": associations,
    }
