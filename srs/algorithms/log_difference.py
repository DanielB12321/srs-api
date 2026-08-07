"""Mean logarithmic difference across the shared elements."""

from math import log10

from .base import PairwiseSimilarity, weighted_mean


class LogDifferenceSimilarity(PairwiseSimilarity):
    """
    Compare concentrations by how many decades apart they sit.

    Identical concentrations score 1 for an element and a tenfold difference
    scores 0.5. Working in logs makes proportional differences comparable across
    elements whose concentrations differ by orders of magnitude, so a 10 ppm gap
    in copper does not drown out a 0.01 ppm gap in gold.

    Normalisation: the per-element score is 1 / (1 + |difference|), averaged
    over the shared elements. That is already bounded to (0, 1] with 1 meaning
    identical, so no extra squashing is applied. The transform is monotone in
    the mean absolute log difference, so rank-based benchmark metrics are
    unaffected by the choice.
    """

    id = "log_difference_similarity"
    version = "1.0.0"
    capabilities = frozenset()

    def score_vectors(self, prepared):
        input_vector = prepared.input_vector
        reference_vector = prepared.reference_vector

        # Preprocessing may already have taken the values into log space, and
        # taking the logarithm of a centred log value is meaningless, so this
        # only applies the transform when nothing else has.
        if not prepared.in_log_space:
            input_vector = [log10(value) for value in input_vector]
            reference_vector = [log10(value) for value in reference_vector]

        element_scores = [
            1 / (1 + abs(left - right))
            for left, right in zip(input_vector, reference_vector)
        ]
        return weighted_mean(element_scores, prepared.weights)
