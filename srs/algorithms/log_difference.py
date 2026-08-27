"""Mean logarithmic difference across the shared elements."""

from math import log10

from .base import PairwiseSimilarity, weighted_mean


class LogDifferenceSimilarity(PairwiseSimilarity):
    """Average similarity from the base-10 difference of shared elements.

    An identical value scores ``1`` and a tenfold difference scores ``0.5``.
    """

    id = "log_difference_similarity"
    version = "1.0.0"
    capabilities = frozenset()

    def score_vectors(self, prepared):
        input_vector = prepared.input_vector
        reference_vector = prepared.reference_vector

        # Do not apply a logarithm when preprocessing already produced log data.
        if not prepared.in_log_space:
            input_vector = [log10(value) for value in input_vector]
            reference_vector = [log10(value) for value in reference_vector]

        element_scores = [
            1 / (1 + abs(left - right))
            for left, right in zip(input_vector, reference_vector)
        ]
        return weighted_mean(element_scores, prepared.weights)
