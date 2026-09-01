"""Base classes and shared calculations for similarity algorithms."""

from abc import ABC, abstractmethod
from time import perf_counter

from ..preprocessing import describe, prepare_vectors, resolve_options
from .envelope import Evidence, Match, RunResult


def weighted_mean(values, weights=None):
    """Return an arithmetic or weighted mean."""
    if weights is None:
        return sum(values) / len(values)

    return sum(
        weight * value
        for weight, value in zip(weights, values)
    ) / sum(weights)


def weighted_dot(left, right, weights=None):
    """Dot product, honouring per-element weights when there are any."""
    if weights is None:
        return sum(
            left_value * right_value
            for left_value, right_value in zip(left, right)
        )

    return sum(
        weight * left_value * right_value
        for weight, left_value, right_value in zip(weights, left, right)
    )


def signed_evidence(prepared, raw_contributions):
    """Scale per-element effects and split them by whether they help the match."""
    weights = prepared.weights or [1.0] * len(raw_contributions)
    weighted = [
        weight * contribution
        for weight, contribution in zip(weights, raw_contributions)
    ]
    total = sum(abs(contribution) for contribution in weighted)

    if not total:
        return [], []

    supporting = []
    conflicting = []
    imputed = prepared.imputed or [False] * len(prepared.symbols)

    for symbol, contribution, was_imputed in zip(
        prepared.symbols,
        weighted,
        imputed,
    ):
        scaled = contribution / total
        entry = Evidence(symbol, scaled, bool(was_imputed))
        if scaled >= 0:
            supporting.append(entry)
        else:
            conflicting.append(entry)

    supporting.sort(key=lambda entry: -entry.contribution)
    conflicting.sort(key=lambda entry: entry.contribution)
    return supporting, conflicting


class SimilarityAlgorithm(ABC):
    """Interface for scoring samples against a reference library."""

    # Registry key sent as ``similarity_method``.
    id: str = ""
    # Bump when score calculations change.
    version: str = "0.0.0"
    # Optional result sections produced by the algorithm.
    capabilities: frozenset = frozenset()

    def raw_scores(self, prepared):
        """Return extra metrics to store beside the similarity score."""
        return {}

    def evidence(self, prepared):
        """Return supporting and conflicting evidence for one comparison."""
        return [], []

    def confidence(self, deposit_id, nearest_deposits):
        """Return confidence from nearby deposits when supported."""
        return None

    @abstractmethod
    def compare(self, samples, references, config=None):
        """
        Score samples against reference signatures and return a RunResult.

        samples is a list of dicts shaped
        {"sample_code": str, "values": {symbol: float}}, optionally carrying
        latitude and longitude. references is any iterable of dicts shaped
        {"id": int, "values": {symbol: float}}, optionally carrying deposit_id,
        deposit_name, and deposit_class. config carries the preprocessing
        options and per-run parameters such as top_n.
        """


class PairwiseSimilarity(SimilarityAlgorithm):
    """Base for algorithms that score one reference at a time."""

    def score_pair(
        self,
        input_values,
        reference_values,
        common_elements,
        preprocessing=None,
        imputed_elements=None,
    ):
        """Prepare and score one sample/reference pair."""
        prepared = prepare_vectors(
            input_values,
            reference_values,
            common_elements,
            self._as_options(preprocessing),
            imputed_elements,
        )

        if not prepared.input_vector:
            return 0

        return self.score_vectors(prepared)

    @staticmethod
    def _as_options(preprocessing):
        """Resolve a request block, or pass through an already-resolved dict."""
        if preprocessing and "handle_missing" in preprocessing:
            return preprocessing

        return resolve_options(preprocessing)

    @abstractmethod
    def score_vectors(self, prepared):
        """Return a similarity in ``[0, 1]`` for prepared vectors."""

    def compare(self, samples, references, config=None):
        """Rank every reference for each supplied sample."""
        config = config or {}
        top_n = int(config.get("top_n", 200))
        # Resolve once before the reference loop.
        options = resolve_options(
            config.get("preprocessing"),
            config.get("selected_elements"),
        )

        # A list can be reused when more than one sample is supplied.
        reference_list = list(references)
        started = perf_counter()

        rankings = [
            self._rank_one_sample(sample, reference_list, options, top_n)
            for sample in samples
        ]
        detail_top_n = max(0, int(config.get("detail_top_n", 10)))
        references_by_id = {
            reference.get("id"): reference
            for reference in reference_list
        }

        # Evidence is deliberately limited to the leading matches so a large
        # complete ranking does not spend time building detail nobody sees.
        for sample, ranking in zip(samples, rankings):
            input_values = sample.get("values") or {}
            input_imputed = set(sample.get("imputed") or ())
            for match in ranking[:detail_top_n]:
                reference = references_by_id.get(match.reference_sample_id)
                reference_values = (reference or {}).get("values") or {}
                common_elements = set(input_values) & set(reference_values)
                if not common_elements:
                    continue

                prepared = prepare_vectors(
                    input_values,
                    reference_values,
                    common_elements,
                    options,
                    input_imputed | set((reference or {}).get("imputed") or ()),
                )
                match.supporting, match.conflicting = self.evidence(prepared)
        runtime_ms = (perf_counter() - started) * 1000

        # Record elements present on both sides of the comparison.
        elements_used = sorted({
            symbol
            for sample in samples
            for symbol in (sample.get("values") or {})
        } & {
            symbol
            for reference in reference_list
            for symbol in (reference.get("values") or {})
        })

        warnings = []
        if not reference_list:
            warnings.append("The reference library is empty; no matches were produced.")

        return RunResult(
            algorithm_id=self.id,
            algorithm_version=self.version,
            algorithm_params={"top_n": top_n},
            matches=rankings[0] if rankings else [],
            runtime_ms=runtime_ms,
            preprocessing=describe(options, elements_used),
            reference_library_version=config.get("reference_library_version", ""),
            dataset_id=config.get("dataset_id"),
            sample_results=[
                {
                    "sample_id": sample.get("sample_code"),
                    "lat": sample.get("latitude"),
                    "lon": sample.get("longitude"),
                    "top_matches": [
                        {
                            "reference_sample_id": match.reference_sample_id,
                            "deposit_id": match.deposit_id,
                            "similarity": float(match.similarity),
                        }
                        for match in ranking[:5]
                    ],
                }
                for sample, ranking in zip(samples, rankings)
            ],
            warnings=warnings,
        )

    def _rank_one_sample(self, sample, references, options, top_n):
        input_values = sample.get("values") or {}
        input_imputed = set(sample.get("imputed") or ())
        scored = []

        for reference in references:
            reference_values = reference.get("values") or {}
            common_elements = set(input_values) & set(reference_values)
            if not common_elements:
                continue

            scored.append((
                self.score_pair(
                    input_values,
                    reference_values,
                    common_elements,
                    options,
                    input_imputed | set(reference.get("imputed") or ()),
                ),
                reference,
            ))

        # Ties break on the higher reference ID, matching how the API's bounded
        # heap orders equal scores. Keeping the two consistent means a match
        # list does not reorder itself depending on which path produced it.
        scored.sort(key=lambda item: (-item[0], -(item[1].get("id") or 0)))

        return [
            Match(
                rank=rank,
                similarity=score,
                reference_sample_id=reference.get("id"),
                deposit_id=reference.get("deposit_id", ""),
                deposit_name=reference.get("deposit_name", ""),
                deposit_class=reference.get("deposit_class", ""),
            )
            for rank, (score, reference) in enumerate(scored[:top_n], start=1)
        ]
