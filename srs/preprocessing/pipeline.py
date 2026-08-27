"""Shared measurement cleanup and vector preparation for every algorithm."""

from dataclasses import dataclass, field
from math import log10

from ..services.units import concentration_to_ppm

# Stored with each run so previous results remain identifiable.
PIPELINE_VERSION = "1.0"

# Ways to handle measurements below their detection limit.
SKIP = "skip"
HALF_DETECTION_LIMIT = "half_dl"
DETECTION_LIMIT = "detection_limit"
CENSORED_POLICIES = frozenset({SKIP, HALF_DETECTION_LIMIT, DETECTION_LIMIT})

# How much each element counts towards the final score.
EQUAL = "equal"
SELECTED_BOOST = "selected_boost"
WEIGHTING_MODES = frozenset({EQUAL, SELECTED_BOOST})

SELECTED_ELEMENT_WEIGHT = 2.0


def normalise_symbol(symbol):
    """Return a consistently capitalised element symbol, so 'CU' becomes 'Cu'."""
    symbol = str(symbol or "").strip()

    if not symbol:
        return ""

    return symbol[0].upper() + symbol[1:].lower()


@dataclass
class PreparedVectors:
    """Two aligned vectors and the metadata needed to score them."""

    symbols: list
    input_vector: list
    reference_vector: list
    # True when the value was substituted rather than measured.
    imputed: list = field(default_factory=list)
    # None keeps the original unweighted calculation.
    weights: list = None
    options: dict = field(default_factory=dict)

    @property
    def in_log_space(self):
        """True when a transform has already taken the values into logs."""
        return bool(
            self.options.get("normalise")
            or self.options.get("log_transform")
        )

    def __len__(self):
        return len(self.symbols)


def resolve_options(preprocessing=None, selected_elements=None):
    """Return a complete and valid preprocessing options dictionary."""
    preprocessing = preprocessing or {}

    censored_policy = preprocessing.get("handle_missing") or SKIP
    if censored_policy not in CENSORED_POLICIES:
        censored_policy = SKIP

    weighting_mode = preprocessing.get("weighting_mode") or EQUAL
    if weighting_mode not in WEIGHTING_MODES:
        weighting_mode = EQUAL

    symbols = [
        normalise_symbol(symbol)
        for symbol in (selected_elements or [])
    ]

    normalise = bool(preprocessing.get("normalise"))

    return {
        "normalise": normalise,
        # CLR already works in log space, so a second log transform is not used.
        "log_transform": bool(preprocessing.get("log_transform")) and not normalise,
        "handle_missing": censored_policy,
        "weighting_mode": weighting_mode,
        "selected_elements": [symbol for symbol in symbols if symbol],
    }


def _read_measurement(measurement):
    """Read the common fields from a request dictionary or model instance."""
    if isinstance(measurement, dict):
        return (
            normalise_symbol(
                measurement.get("element_symbol") or measurement.get("symbol")
            ),
            measurement.get("value"),
            measurement.get("unit"),
            bool(measurement.get("below_detection_limit")),
            measurement.get("detection_limit"),
        )

    return (
        measurement.element.symbol,
        measurement.value,
        measurement.unit,
        measurement.below_detection_limit,
        measurement.detection_limit,
    )


def _censored_value(detection_limit, unit, censored_policy):
    """Return the substituted ppm value for a censored measurement, or None."""
    if censored_policy == SKIP:
        return None

    limit = concentration_to_ppm(detection_limit, unit)
    if limit is None:
        return None

    if censored_policy == HALF_DETECTION_LIMIT:
        # Half the limit avoids introducing zero, which cannot be logged.
        return limit / 2

    return limit


def extract_values(measurements, options=None):
    """Return ppm values and the symbols whose values were substituted."""
    options = options or resolve_options()
    censored_policy = options.get("handle_missing", SKIP)

    # Boost mode keeps unselected elements so the selected ones can be weighted.
    selected = set(options.get("selected_elements") or [])
    restrict_to_selected = bool(selected) and (
        options.get("weighting_mode") != SELECTED_BOOST
    )

    values = {}
    imputed = set()

    for measurement in measurements:
        symbol, value, unit, below_limit, detection_limit = _read_measurement(
            measurement
        )

        if not symbol:
            continue

        if restrict_to_selected and symbol not in selected:
            continue

        if below_limit:
            substituted = _censored_value(detection_limit, unit, censored_policy)
            if substituted is None:
                continue

            values[symbol] = substituted
            imputed.add(symbol)
            continue

        converted = concentration_to_ppm(value, unit)
        if converted is not None:
            values[symbol] = converted

    return values, imputed


def _centred_logs(values):
    """Centre log values on their mean, which is the CLR transform."""
    logs = [log10(value) for value in values]
    mean = sum(logs) / len(logs)
    return [value - mean for value in logs]


def _weights_for(symbols, options):
    """Return element weights, or ``None`` when all weights are equal."""
    if options.get("weighting_mode") != SELECTED_BOOST:
        return None

    selected = set(options.get("selected_elements") or [])
    if not selected:
        return None

    weights = [
        SELECTED_ELEMENT_WEIGHT if symbol in selected else 1.0
        for symbol in symbols
    ]

    if len(set(weights)) < 2:
        return None

    return weights


def prepare_vectors(
    input_values,
    reference_values,
    common_elements,
    options=None,
    imputed_elements=None,
):
    """Build aligned vectors from the elements shared by two samples."""
    options = options or resolve_options()
    imputed_elements = imputed_elements or set()
    symbols = sorted(common_elements)

    input_vector = [input_values[symbol] for symbol in symbols]
    reference_vector = [reference_values[symbol] for symbol in symbols]

    if options.get("normalise"):
        # CLR keeps relative element patterns rather than overall concentration.
        input_vector = _centred_logs(input_vector)
        reference_vector = _centred_logs(reference_vector)
    elif options.get("log_transform"):
        input_vector = [log10(value) for value in input_vector]
        reference_vector = [log10(value) for value in reference_vector]

    return PreparedVectors(
        symbols=symbols,
        input_vector=input_vector,
        reference_vector=reference_vector,
        imputed=[symbol in imputed_elements for symbol in symbols],
        weights=_weights_for(symbols, options),
        options=options,
    )


def describe(options, symbols=None):
    """Describe the preprocessing settings stored with an analysis run."""
    options = options or resolve_options()
    symbols = list(symbols or [])

    return {
        "pipeline_version": PIPELINE_VERSION,
        "censored_policy": options.get("handle_missing", SKIP),
        "normalise": bool(options.get("normalise")),
        "log_transform": bool(options.get("log_transform")),
        "weighting_mode": options.get("weighting_mode", EQUAL),
        "selected_elements": list(options.get("selected_elements") or []),
        "elements_used": symbols,
        "n_shared_elements": len(symbols),
    }
