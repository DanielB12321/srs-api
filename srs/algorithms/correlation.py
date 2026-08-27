"""Pearson correlation between two element patterns."""

from .base import PairwiseSimilarity, weighted_dot, weighted_mean
from .log_difference import LogDifferenceSimilarity


class CorrelationSimilarity(PairwiseSimilarity):
    """Compare shared element patterns using Pearson correlation.

    Pearson's ``-1`` to ``1`` range is mapped onto ``[0, 1]``. Constant vectors
    score zero because their correlation is undefined.
    """

    id = "correlation"
    version = "1.0.0"
    capabilities = frozenset()

    def score_vectors(self, prepared):
        input_vector = prepared.input_vector
        reference_vector = prepared.reference_vector
        weights = prepared.weights

        # Keep the existing log-difference fallback for one shared element.
        if len(input_vector) < 2:
            return LogDifferenceSimilarity().score_vectors(prepared)

        input_mean = weighted_mean(input_vector, weights)
        reference_mean = weighted_mean(reference_vector, weights)
        input_centred = [value - input_mean for value in input_vector]
        reference_centred = [value - reference_mean for value in reference_vector]

        numerator = weighted_dot(input_centred, reference_centred, weights)
        denominator = (
            weighted_dot(input_centred, input_centred, weights)
            * weighted_dot(reference_centred, reference_centred, weights)
        ) ** 0.5

        return (1 + numerator / denominator) / 2 if denominator else 0
