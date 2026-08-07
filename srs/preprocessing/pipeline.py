"""
The shared preprocessing head.

Every algorithm receives values that came through here, which is what makes a
benchmark compare algorithms rather than pipelines. No algorithm gets to
preprocess more cleverly than its rivals, so a win on the leaderboard is a win
for the algorithm.

Two stages, deliberately separate:

* extract_values() turns raw measurements into a symbol-to-ppm mapping. Unit
  conversion, the censored-data policy, and the element selection live here.
* prepare_vectors() turns two such mappings into aligned numeric vectors, the
  imputed mask, and per-element weights.

Defaults reproduce the behaviour that existed before this module. Every
transform is opt-in through the request's preprocessing block, so an existing
client sees identical scores.
"""

from dataclasses import dataclass, field
from math import log10

from ..services.units import concentration_to_ppm

#: Bumped whenever a transform changes in a way that moves scores. Stored with
#: every run so an old result stays interpretable.
PIPELINE_VERSION = "1.0"

# How to treat a measurement reported as below the detection limit. Roughly 15%
# of the reference library is censored this way, and every censored row carries
# its detection limit, so substitution is always possible.
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
    """
    Two aligned, transformed vectors plus everything an algorithm needs about
    them.

    Algorithms receive this object rather than a handful of positional
    arguments, so a field can be added later without breaking any algorithm
    written against the current interface. That matters when algorithms arrive
    from different people at different times.
    """

    symbols: list
    input_vector: list
    reference_vector: list
    #: True where the value was substituted rather than measured. This is the
    #: only source for the imputed flag on evidence entries.
    imputed: list = field(default_factory=list)
    #: Per-element weights, or None when every element counts equally. None is
    #: not the same as a list of ones: it selects the unweighted arithmetic, so
    #: an equally weighted run is bit-for-bit what it was before weighting
    #: existed.
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
    """
    Validate the request's preprocessing block into a complete options dict.

    Unrecognised policy names fall back to the default rather than raising. The
    preprocessing block is free-form JSON sent by a separate frontend, so a
    typo or a newer client's unknown value must never take an analysis down.
    """
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

    return {
        "normalise": bool(preprocessing.get("normalise")),
        "log_transform": bool(preprocessing.get("log_transform")),
        "handle_missing": censored_policy,
        "weighting_mode": weighting_mode,
        "selected_elements": [symbol for symbol in symbols if symbol],
    }


def _read_measurement(measurement):
    """
    Read one measurement from either a request dict or a database row.

    Input samples arrive as JSON dicts and reference samples as model
    instances. Normalising both here keeps the policy logic below written once.
    """
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
        # Substituting half the detection limit is the standard compromise: the
        # true value lies somewhere in [0, limit], and zero cannot be logged.
        return limit / 2

    return limit


def extract_values(measurements, options=None):
    """
    Turn raw measurements into a symbol-to-ppm mapping.

    Returns the mapping and the set of symbols whose value was substituted
    rather than measured. Values that cannot be compared honestly, such as an
    unsupported unit or a non-positive concentration, are left out entirely
    rather than being coerced onto the wrong scale.
    """
    options = options or resolve_options()
    censored_policy = options.get("handle_missing", SKIP)

    # A hard filter, unless the weighting mode says to treat the selection as a
    # preference. Boosting selected elements would be meaningless if everything
    # else had already been discarded.
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
    """
    Return per-element weights, or None when they would all be equal.

    Returning None rather than a list of ones is deliberate: it routes scoring
    through the original unweighted arithmetic, so enabling this feature cannot
    perturb a run that does not use it.
    """
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
    """
    Turn two symbol-to-value mappings into aligned, transformed vectors.

    Sorting the symbols is what keeps the two vectors aligned element by
    element, and it makes the result independent of the order measurements
    happened to arrive in.
    """
    options = options or resolve_options()
    imputed_elements = imputed_elements or set()
    symbols = sorted(common_elements)

    input_vector = [input_values[symbol] for symbol in symbols]
    reference_vector = [reference_values[symbol] for symbol in symbols]

    if options.get("normalise"):
        # CLR centres each sample on its own geometric mean, removing overall
        # concentration and leaving the relative pattern.
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
    """
    Summarise the pipeline configuration for the run block of the envelope.

    This is what makes a stored result reproducible: the pipeline version plus
    the exact policies and element suite that produced it.
    """
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
