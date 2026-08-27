"""Cosine similarity between two element patterns."""

from .base import PairwiseSimilarity, weighted_dot


class AssociationSimilarity(PairwiseSimilarity):
    """Compare vector direction using cosine similarity.

    Cosine's ``-1`` to ``1`` range is mapped and clamped to ``[0, 1]``.
    """

    id = "association"
    version = "1.0.0"
    capabilities = frozenset()

    def score_vectors(self, prepared):
        input_vector = prepared.input_vector
        reference_vector = prepared.reference_vector
        weights = prepared.weights

        numerator = weighted_dot(input_vector, reference_vector, weights)
        denominator = (
            weighted_dot(input_vector, input_vector, weights)
            * weighted_dot(reference_vector, reference_vector, weights)
        ) ** 0.5
        cosine = numerator / denominator if denominator else 0

        return max(0, min(1, (cosine + 1) / 2))
