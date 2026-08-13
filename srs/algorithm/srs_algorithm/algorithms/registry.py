"""
Registry mapping algorithm id -> instance.

The API and benchmark harness both go through get_algorithm() rather than
importing a class directly, so a new algorithm becomes available everywhere
the moment it's added here — no other file needs to change.
"""

from .aitchison import KnnAitchisonSimilarity
from .association import AssociationSimilarity
from .correlation import CorrelationSimilarity
from .distance import DistanceSimilarity
from .log_difference import LogDifferenceSimilarity

# Add one line here per new algorithm. Each id must match what clients send
# as similarity_method.
ALGORITHMS = {
    algorithm.id: algorithm
    for algorithm in (
        KnnAitchisonSimilarity(),
        AssociationSimilarity(),
        CorrelationSimilarity(),
        DistanceSimilarity(),
        LogDifferenceSimilarity(),
    )
}


def get_algorithm(algorithm_id):
    """Look up a fitted algorithm instance by its registry id."""
    try:
        return ALGORITHMS[algorithm_id]
    except KeyError:
        raise ValueError(
            f"Unknown similarity_method {algorithm_id!r}; "
            f"choose from {sorted(ALGORITHMS)}"
        )


def list_algorithms():
    """Return {id: capabilities} for every registered algorithm — useful for
    an API endpoint that lets the frontend discover what's available."""
    return {
        algorithm_id: sorted(algorithm.capabilities)
        for algorithm_id, algorithm in ALGORITHMS.items()
    }
