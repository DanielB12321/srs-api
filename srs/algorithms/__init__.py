"""Registry of similarity algorithms available to the API."""

from django.conf import settings

from .association import AssociationSimilarity
from .base import PairwiseSimilarity, SimilarityAlgorithm
from .correlation import CorrelationSimilarity
from .distance import DistanceSimilarity
from .envelope import Evidence, Match, RunResult, SCHEMA_VERSION, validate_envelope
from .knn_aitchison import KnnAitchisonSimilarity
from .log_difference import LogDifferenceSimilarity
from .ml_ensemble import XgbSvmEnsembleSimilarity

# Used when no valid default is configured.
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
    """Return the selected algorithm, falling back to the configured default."""
    algorithm_class = ALGORITHMS.get(algorithm_id)

    if algorithm_class is None:
        algorithm_class = ALGORITHMS[default_algorithm_id()]

    # Instances do not share state between background analysis threads.
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
