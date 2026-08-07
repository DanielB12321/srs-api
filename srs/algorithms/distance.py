"""Mean absolute difference across the shared elements."""

from .base import PairwiseSimilarity, weighted_mean


class DistanceSimilarity(PairwiseSimilarity):
    """
    Compare samples on whatever scale preprocessing left them on.

    This applies no transform of its own, which makes it the most direct read of
    what the preprocessing head produced. On raw concentrations that means large
    numbers dominate and a 900 ppm gap in copper swamps everything else, so this
    method is really only meaningful with a log or CLR transform enabled. With
    log_transform it is arithmetically identical to log_difference_similarity,
    and with CLR it is a per-element Aitchison difference, which is the
    relationship the compositional work builds on.

    Normalisation: the per-element score is 1 / (1 + |difference|), averaged
    over the shared elements. Already bounded to (0, 1] with 1 meaning
    identical, and monotone in the mean absolute difference.
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
