"""Compositional similarity: Aitchison distance in CLR space, with Spearman rho."""

from math import log10, sqrt

from ..preprocessing import prepare_vectors, resolve_options
from .base import PairwiseSimilarity, weighted_dot
from .envelope import Evidence

#: How many nearest references vote on a match's confidence.
DEFAULT_K = 5

#: Evidence, raw metrics, and confidence are attached to this many matches.
#: Computing them for a full ranking of every reference would multiply the row
#: size of a result set that already runs to hundreds of rows per sample.
DEFAULT_DETAIL_TOP_N = 10

_LEVEL_THRESHOLDS = ((0.6, "high"), (0.3, "medium"))


def _average_ranks(values):
    """Rank values from 1, giving tied values their average rank."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0

    while position < len(order):
        last = position
        while (
            last + 1 < len(order)
            and values[order[last + 1]] == values[order[position]]
        ):
            last += 1

        shared = (position + last) / 2 + 1
        for index in range(position, last + 1):
            ranks[order[index]] = shared

        position = last + 1

    return ranks


def _pearson(left, right):
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_centred = [value - left_mean for value in left]
    right_centred = [value - right_mean for value in right]

    numerator = sum(a * b for a, b in zip(left_centred, right_centred))
    denominator = (
        sum(value * value for value in left_centred)
        * sum(value * value for value in right_centred)
    ) ** 0.5

    return numerator / denominator if denominator else 0.0


def _centred(values):
    mean = sum(values) / len(values)
    return [value - mean for value in values]


class KnnAitchisonSimilarity(PairwiseSimilarity):
    """
    Compare samples as compositions rather than as lists of concentrations.

    Geochemical measurements are compositional: only the ratios between
    elements carry information, because the absolute values depend on how much
    of the sample was dissolved and how it was reported. The centred log-ratio
    transform is the standard way to work in that space, and the Aitchison
    distance is the Euclidean distance between two CLR vectors. Two samples
    with the same element ratios sit at distance zero however concentrated
    either one is.

    This applies the CLR itself when preprocessing has not, so the algorithm is
    correct whichever transform the request selected. Closure, rescaling a
    composition to a constant sum, is deliberately not applied: the CLR centres
    on the geometric mean and is therefore already scale-invariant, so closure
    before it would be an exact no-op.

    Spearman's rho is reported alongside as a raw metric. It measures whether
    the two samples order their elements the same way, which is a different
    question from how far apart they sit, so a pair can be distant yet strongly
    rank-correlated.

    Normalisation: the Aitchison distance is unbounded above, and is mapped to
    a similarity with 1 / (1 + distance). That is monotone decreasing, so
    ranking and rank-based benchmark metrics are unaffected by the choice, and
    bounded to (0, 1] with 1 meaning compositionally identical.

    Be aware that these similarities are much lower in absolute terms than the
    concentration-based methods produce, because Aitchison distances across a
    diverse reference library are large. A user interface threshold tuned
    against log_difference_similarity will look broken here. Compare
    confidence.level between algorithms, never the raw score.
    """

    id = "knn_aitchison"
    version = "1.0.0"
    capabilities = frozenset({"evidence"})

    def score_vectors(self, prepared):
        left, right = self._clr_pair(prepared)
        differences = [a - b for a, b in zip(left, right)]

        # Weighted Euclidean when weights are present, plain Euclidean when
        # they are not, which is what weighted_dot handles for us.
        distance = sqrt(weighted_dot(differences, differences, prepared.weights))

        return 1 / (1 + distance)

    def _clr_pair(self, prepared):
        """
        Return both vectors in CLR space, whatever preprocessing already did.

        The user picks the algorithm and the preprocessing independently, so
        this cannot assume the request happened to enable CLR.
        """
        options = prepared.options

        if options.get("normalise"):
            # Already centred log-ratios.
            return prepared.input_vector, prepared.reference_vector

        if options.get("log_transform"):
            # Logs, but not yet centred.
            return _centred(prepared.input_vector), _centred(prepared.reference_vector)

        return (
            _centred([log10(value) for value in prepared.input_vector]),
            _centred([log10(value) for value in prepared.reference_vector]),
        )

    def raw_scores(self, prepared):
        """Return the algorithm-native metrics that sit beside the similarity."""
        left, right = self._clr_pair(prepared)
        differences = [a - b for a, b in zip(left, right)]
        distance = sqrt(weighted_dot(differences, differences, prepared.weights))

        # Rank correlation is left unweighted. A weighted rank correlation is
        # not a standard quantity, and inventing one would make the reported
        # metric hard to compare with anything published.
        rho = (
            _pearson(_average_ranks(left), _average_ranks(right))
            if len(left) >= 2
            else 0.0
        )

        return {"aitchison_distance": distance, "spearman_rho": rho}

    def evidence(self, prepared):
        """
        Split the distance into per-element contributions.

        Every element adds its squared CLR difference to the total. An element
        that differs less than the average pulls the pair together and is
        reported as supporting; one that differs more pushes them apart and is
        reported as conflicting. Contributions are scaled to sum to one in
        absolute value so they can be read as shares of the disagreement.
        """
        left, right = self._clr_pair(prepared)
        squared = [(a - b) ** 2 for a, b in zip(left, right)]
        mean_squared = sum(squared) / len(squared)

        raw = [mean_squared - value for value in squared]
        total = sum(abs(value) for value in raw)

        supporting = []
        conflicting = []

        for symbol, contribution, imputed in zip(
            prepared.symbols,
            raw,
            prepared.imputed or [False] * len(prepared.symbols),
        ):
            scaled = contribution / total if total else 0.0
            entry = Evidence(element=symbol, contribution=scaled, imputed=imputed)

            if scaled >= 0:
                supporting.append(entry)
            else:
                conflicting.append(entry)

        # Strongest first on each side, so a consumer showing only the top few
        # shows the ones that actually moved the result.
        supporting.sort(key=lambda item: -item.contribution)
        conflicting.sort(key=lambda item: item.contribution)

        return supporting, conflicting

    def compare(self, samples, references, config=None):
        """
        Rank as usual, then attach the detail only this algorithm can produce.

        Raw metrics, evidence, and kNN confidence are computed for the leading
        matches rather than for every reference. Ranking already used all of
        them, so nothing is lost from the ordering; what is avoided is
        attaching several blocks of detail to hundreds of rows that no one
        opens.
        """
        config = config or {}
        # Materialised before ranking so the references can be revisited below
        # even when a generator was passed in.
        reference_list = list(references)
        result = super().compare(samples, reference_list, config)

        if not result.matches or not samples:
            return result

        neighbours = int(config.get("k", DEFAULT_K))
        detail_top_n = int(config.get("detail_top_n", DEFAULT_DETAIL_TOP_N))
        result.algorithm_params.update({"k": neighbours})

        options = resolve_options(
            config.get("preprocessing"),
            config.get("selected_elements"),
        )
        sample = samples[0]
        input_values = sample.get("values") or {}
        input_imputed = set(sample.get("imputed") or ())
        references_by_id = {
            reference.get("id"): reference
            for reference in reference_list
        }

        # Which deposits the k nearest references belong to. This is the kNN
        # part: a match is more trustworthy when several samples from the same
        # deposit independently rank near the top.
        nearest_deposits = [
            match.deposit_id or match.deposit_name
            for match in result.matches[:neighbours]
        ]

        for match in result.matches[:detail_top_n]:
            reference = references_by_id.get(match.reference_sample_id)
            if reference is None:
                continue

            reference_values = reference.get("values") or {}
            common_elements = set(input_values) & set(reference_values)
            if not common_elements:
                continue

            prepared = prepare_vectors(
                input_values,
                reference_values,
                common_elements,
                options,
                input_imputed | set(reference.get("imputed") or ()),
            )

            match.scores.update(self.raw_scores(prepared))
            match.supporting, match.conflicting = self.evidence(prepared)
            match.confidence = self.confidence(
                match.deposit_id or match.deposit_name,
                nearest_deposits,
            )

        return result

    def confidence(self, deposit_id, nearest_deposits):
        """Judge a match by how many of the nearest references agree with it."""
        deposit = deposit_id
        agreeing = sum(
            1
            for neighbour in nearest_deposits
            if neighbour and neighbour == deposit
        )
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
