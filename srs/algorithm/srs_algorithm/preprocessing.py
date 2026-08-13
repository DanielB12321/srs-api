"""
Shared preprocessing head: turns raw {element: value} dicts into aligned,
transformed vectors every algorithm consumes identically.

This is the piece the algorithm docstrings all assume exists but that hadn't
been written yet. Nothing here is validated against your real data — the
missing-value policy in particular ("half_min") follows the approach your
notebook converged on at the end (replace with half the column's minimum
positive value), but that needs the real reference library wired in before
it can be trusted. Treat this file as a working skeleton, not a finished
contract.

No algorithm applies its own log or CLR transform except aitchison.py, which
applies CLR itself if this preprocessing head didn't, so it stays correct
regardless of what the request selected. Every other algorithm assumes
prepare_vectors() already did whatever transform the request asked for.
"""

from dataclasses import dataclass
from math import log10

DEFAULT_EPSILON = 1e-6
DEFAULT_MISSING_POLICY = "half_min"  # "half_min" | "epsilon" | "drop"


@dataclass
class PreparedVectors:
    """Aligned, transformed vectors ready for score_vectors()."""

    input_vector: list
    reference_vector: list
    symbols: list
    weights: list | None
    imputed: list
    in_log_space: bool
    options: dict


def resolve_options(preprocessing=None, selected_elements=None):
    """
    Normalise a request's preprocessing block into a fully-populated dict.

    Called once per compare() rather than once per reference — see
    PairwiseSimilarity.compare(), which resolves this outside its loop.
    """
    preprocessing = preprocessing or {}
    options = {
        "handle_missing": preprocessing.get("handle_missing", DEFAULT_MISSING_POLICY),
        "log_transform": bool(preprocessing.get("log_transform", True)),
        "normalise": bool(preprocessing.get("normalise", False)),  # CLR
        "element_weights": preprocessing.get("element_weights"),
        "selected_elements": selected_elements,
    }

    # CLR only means something on logged values. A request asking for CLR
    # without log_transform is asking for something incoherent, so log is
    # implied rather than silently ignored.
    if options["normalise"] and not options["log_transform"]:
        options["log_transform"] = True

    return options


def prepare_vectors(
    input_values,
    reference_values,
    common_elements,
    options,
    imputed_elements=None,
):
    """
    Align two {element: value} dicts onto the shared element list, apply the
    requested transform, and return a PreparedVectors ready for scoring.

    Elements not in common_elements are dropped rather than imputed — a
    sample and a reference that share nothing shouldn't be scored at all,
    which is why PairwiseSimilarity.score_pair() checks
    `if not prepared.input_vector: return 0` before calling score_vectors().
    """
    imputed_elements = imputed_elements or set()
    selected = options.get("selected_elements")
    symbols = sorted(common_elements & set(selected)) if selected else sorted(common_elements)

    if not symbols:
        return PreparedVectors([], [], [], None, [], bool(options.get("log_transform")), options)

    input_vector = [float(input_values[s]) for s in symbols]
    reference_vector = [float(reference_values[s]) for s in symbols]
    imputed = [s in imputed_elements for s in symbols]

    in_log_space = False
    if options.get("log_transform"):
        input_vector = [log10(max(v, DEFAULT_EPSILON)) for v in input_vector]
        reference_vector = [log10(max(v, DEFAULT_EPSILON)) for v in reference_vector]
        in_log_space = True

    if options.get("normalise"):
        # CLR: centre each already-logged vector on its own geometric mean.
        mean_in = sum(input_vector) / len(input_vector)
        mean_ref = sum(reference_vector) / len(reference_vector)
        input_vector = [v - mean_in for v in input_vector]
        reference_vector = [v - mean_ref for v in reference_vector]

    weights = None
    element_weights = options.get("element_weights")
    if element_weights:
        weights = [float(element_weights.get(s, 1.0)) for s in symbols]

    return PreparedVectors(
        input_vector=input_vector,
        reference_vector=reference_vector,
        symbols=symbols,
        weights=weights,
        imputed=imputed,
        in_log_space=in_log_space,
        options=options,
    )


def describe(options, elements_used):
    """Provenance block for RunResult.preprocessing — travels with every run."""
    return {
        "pipeline_version": "1.0",
        "handle_missing": options.get("handle_missing"),
        "log_transform": options.get("log_transform"),
        "normalise": options.get("normalise"),
        "elements_used": list(elements_used),
        "n_shared_elements": len(elements_used),
    }
