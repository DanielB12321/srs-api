"""
Shared k-NN vote confidence, usable by any PairwiseSimilarity subclass.

Your notebook validation found per-sample k-NN voting outperformed every
centroid-based approach it tried. This is that finding, factored out of
KnnAitchisonSimilarity so association/correlation/distance/log_difference get
the same confidence.level field without each reimplementing it.
"""

_LEVEL_THRESHOLDS = ((0.6, "high"), (0.3, "medium"))


class VotedConfidenceMixin:
    """
    Judge one match by how many of the k nearest references share its deposit.

    Mix this in before PairwiseSimilarity, e.g.
    `class Foo(VotedConfidenceMixin, PairwiseSimilarity)`. It only reads the
    ranking compare() already produced, so it costs nothing extra to compute.
    """

    def confidence(self, deposit_id, nearest_deposits):
        agreeing = sum(1 for neighbour in nearest_deposits if neighbour and neighbour == deposit_id)
        consistency = agreeing / len(nearest_deposits) if nearest_deposits else 0.0

        level = "low"
        for threshold, name in _LEVEL_THRESHOLDS:
            if consistency >= threshold:
                level = name
                break

        return {
            "consistency": consistency,
            "n_reference_samples": agreeing,
            "level": level,
        }
