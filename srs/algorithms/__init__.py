"""
The similarity algorithm registry.

Adding an algorithm is two steps: write a module in this package exposing a
SimilarityAlgorithm subclass, then add it to ALGORITHMS below. Nothing else in
the codebase needs to know it exists. The API, the management commands, and the
benchmark harness all reach algorithms through this registry.

Which algorithm runs is decided in one of two places:

* per request, by sending similarity_method in the analysis payload, which is
  how the app offers the choice to a user, and
* per deployment, by setting SRS_DEFAULT_ALGORITHM in settings or the
  environment, which is how the default is changed in code without touching any
  algorithm.

An unrecognised or missing name resolves to the default rather than failing, so
older clients that send nothing keep working.
"""

from django.conf import settings

from .association import AssociationSimilarity
from .base import PairwiseSimilarity, SimilarityAlgorithm
from .correlation import CorrelationSimilarity
from .distance import DistanceSimilarity
from .envelope import Evidence, Match, RunResult, SCHEMA_VERSION, validate_envelope
from .knn_aitchison import KnnAitchisonSimilarity
from .log_difference import LogDifferenceSimilarity
from .ml_ensemble import XgbSvmEnsembleSimilarity

# The last-resort default. This one is used when settings name an algorithm
# that does not exist, so the service still starts on a typo.
FALLBACK_ALGORITHM_ID = "log_difference_similarity"

ALGORITHMS = {
    algorithm.id: algorithm
    for algorithm in (
        LogDifferenceSimilarity,
        CorrelationSimilarity,
        AssociationSimilarity,
        DistanceSimilarity,
        KnnAitchisonSimilarity,
        XgbSvmEnsembleSimilarity,
    )
}


def default_algorithm_id():
    """Return the configured default, falling back if it names nothing real."""
    configured = getattr(settings, "SRS_DEFAULT_ALGORITHM", FALLBACK_ALGORITHM_ID)

    if configured in ALGORITHMS:
        return configured

    return FALLBACK_ALGORITHM_ID


def get_algorithm(algorithm_id=None):
    """
    Return a ready-to-use algorithm instance for the given id.

    Unknown and missing ids resolve to the configured default instead of
    raising, because the analysis endpoint has always accepted a request that
    names no method and clients depend on that.
    """
    algorithm_class = ALGORITHMS.get(algorithm_id)

    if algorithm_class is None:
        algorithm_class = ALGORITHMS[default_algorithm_id()]

    # Algorithms hold no per-run state, so a fresh instance each time keeps
    # them safe to use from the background analysis threads.
    return algorithm_class()


def available_algorithms():
    """Describe every registered algorithm, for a picker in the application."""
    return [
        {
            "id": algorithm.id,
            "version": algorithm.version,
            "name": algorithm.__name__,
            "capabilities": sorted(algorithm.capabilities),
            "description": (algorithm.__doc__ or "").strip().split("\n\n")[0],
            "is_default": algorithm.id == default_algorithm_id(),
        }
        for algorithm in ALGORITHMS.values()
    ]


__all__ = [
    "ALGORITHMS",
    "FALLBACK_ALGORITHM_ID",
    "Evidence",
    "Match",
    "PairwiseSimilarity",
    "RunResult",
    "SCHEMA_VERSION",
    "SimilarityAlgorithm",
    "available_algorithms",
    "default_algorithm_id",
    "get_algorithm",
    "validate_envelope",
]
