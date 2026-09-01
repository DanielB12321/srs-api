"""Mean logarithmic difference across the shared elements."""

from math import log10

from .base import PairwiseSimilarity, signed_evidence, weighted_mean


class LogDifferenceSimilarity(PairwiseSimilarity):
    """Average similarity from the base-10 difference of shared elements.

    An identical value scores ``1`` and a tenfold difference scores ``0.5``.
    """

    id = "log_difference_similarity"
    version = "1.1.0"
    capabilities = frozenset({"evidence"})

    def element_scores(self, prepared):
        """Return the individual element similarities used by the final mean."""
        input_vector = prepared.input_vector
        reference_vector = prepared.reference_vector

        if not prepared.in_log_space:
            input_vector = [log10(value) for value in input_vector]
            reference_vector = [log10(value) for value in reference_vector]

        return [
            1 / (1 + abs(left - right))
            for left, right in zip(input_vector, reference_vector)
        ]

    def score_vectors(self, prepared):
        return weighted_mean(self.element_scores(prepared), prepared.weights)

    def evidence(self, prepared):
        """Show which elements are closer or further apart than a tenfold gap."""
        # A score of 0.5 represents a tenfold concentration difference and is
        # used as the neutral point for an understandable signed contribution.
        effects = [score - 0.5 for score in self.element_scores(prepared)]
        return signed_evidence(prepared, effects)
