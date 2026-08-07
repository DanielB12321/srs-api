"""Pearson correlation between two element patterns."""

from .base import PairwiseSimilarity, weighted_dot, weighted_mean
from .log_difference import LogDifferenceSimilarity


class CorrelationSimilarity(PairwiseSimilarity):
    """
    Score how similarly two samples rise and fall across their shared elements.

    This measures pattern rather than magnitude. Two samples enriched in the
    same elements score highly even when one is ten times more concentrated
    throughout, which suits comparing an ore sample with a weaker halo sample
    from the same system.

    Normalisation: Pearson's r runs from -1 to 1 and is mapped with (1 + r) / 2,
    a monotone transform onto [0, 1], so 0.5 means uncorrelated rather than
    moderately similar. A constant vector has no correlation to measure and
    scores 0.
    """

    id = "correlation"
    version = "1.0.0"
    capabilities = frozenset()

    def score_vectors(self, prepared):
        input_vector = prepared.input_vector
        reference_vector = prepared.reference_vector
        weights = prepared.weights

        # Correlation is undefined for a single point. The original
        # implementation fell through to the log-difference score in that case
        # rather than failing, and callers depend on that.
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
