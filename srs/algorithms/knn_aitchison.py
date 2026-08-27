"""Aitchison similarity with Spearman correlation and match evidence."""

from math import log10, sqrt

from ..preprocessing import prepare_vectors, resolve_options
from .base import PairwiseSimilarity, weighted_dot
from .envelope import Evidence

# Number of nearby references used for confidence.
DEFAULT_K = 5

# Detailed evidence is stored only for the leading matches.
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
    """Compare element ratios using distance in centred log-ratio space.

    Similarity is ``1 / (1 + distance)``. Spearman correlation is included as a
    separate measure of whether the two samples order their elements similarly.
    """

    id = "knn_aitchison"
    version = "1.0.1"
    capabilities = frozenset({"evidence"})

    def score_vectors(self, prepared):
        left, right = self._clr_pair(prepared)
        differences = [a - b for a, b in zip(left, right)]

        # ``weighted_dot`` keeps selected-element weights in the distance.
        distance = sqrt(weighted_dot(differences, differences, prepared.weights))

        return 1 / (1 + distance)

    def _clr_pair(self, prepared):
        """Return both vectors in CLR space without applying a transform twice."""
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

        # Spearman correlation remains unweighted because weighted ranks are not
        # part of the score and would be difficult to interpret.
        rho = (
            _pearson(_average_ranks(left), _average_ranks(right))
            if len(left) >= 2
            else 0.0
        )

        return {"aitchison_distance": distance, "spearman_rho": rho}

    def evidence(self, prepared):
        """Split weighted CLR disagreement into supporting and conflicting elements."""
        left, right = self._clr_pair(prepared)
        squared = [(a - b) ** 2 for a, b in zip(left, right)]
        weights = prepared.weights or [1.0] * len(squared)
        weighted_squared = [
            weight * value
            for weight, value in zip(weights, squared)
        ]
        mean_squared = sum(weighted_squared) / len(weighted_squared)

        raw = [mean_squared - value for value in weighted_squared]
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

        # Keep the strongest entries first for compact result displays.
        supporting.sort(key=lambda item: -item.contribution)
        conflicting.sort(key=lambda item: item.contribution)

        return supporting, conflicting

    def compare(self, samples, references, config=None):
        """Rank references and add detail to the leading matches."""
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

        # Confidence rises when nearby references belong to the same deposit.
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
