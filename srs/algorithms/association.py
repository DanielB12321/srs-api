"""Cosine similarity between two element patterns."""

from .base import PairwiseSimilarity, weighted_dot


class AssociationSimilarity(PairwiseSimilarity):
    """
    Score two samples by the angle between their element vectors.

    Cosine similarity ignores overall magnitude and looks only at direction, so
    it answers "is this the same association of elements" rather than "is this
    the same concentration". Unlike correlation it does not centre the vectors
    first, so a shared baseline of common elements still counts towards the
    score.

    Normalisation: cosine runs from -1 to 1 and is mapped with (cosine + 1) / 2
    onto [0, 1], then clamped. On raw positive concentrations the cosine cannot
    go negative, so scores sit in [0.5, 1] in practice; after CLR centring the
    full range is available. Because the clamp compares against integer bounds,
    an exact match returns integer 1 rather than 1.0, which the envelope coerces
    to a float.
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
