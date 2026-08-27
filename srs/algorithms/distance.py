"""Mean absolute difference across the shared elements."""

from .base import PairwiseSimilarity, weighted_mean


class DistanceSimilarity(PairwiseSimilarity):
    """Average ``1 / (1 + absolute difference)`` over shared elements.

    The calculation uses the scale produced by preprocessing and returns a
    value in ``(0, 1]``.
    """

    id = "distance"
    version = "1.0.0"
    capabilities = frozenset()

    def score_vectors(self, prepared):
        element_scores = [
            1 / (1 + abs(left - right))
            for left, right in zip(prepared.input_vector, prepared.reference_vector)
        ]
        return weighted_mean(element_scores, prepared.weights)
