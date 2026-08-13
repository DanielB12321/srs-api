"""
The interface every similarity algorithm implements.

An algorithm is a plain object. It never receives a request, imports from
Django REST Framework, or writes to the database, so the same class runs inside
the API, a management command, and the benchmark harness without modification.

Preprocessing is deliberately not the algorithm's job. Every algorithm is
handed values that the shared preprocessing head has already transformed, which
is what makes a benchmark compare algorithms rather than pipelines.

Additions in this version, both additive and safe for algorithms that don't
opt in:
  - PairwiseSimilarity.compare() now fills match.confidence for the top
    detail_top_n matches by calling self.confidence(). The default
    confidence() (declared below) returns None, so an algorithm that doesn't
    mix in VotedConfidenceMixin behaves exactly as before. One that does
    (association, correlation, distance, log_difference) gets it for free
    instead of duplicating KnnAitchisonSimilarity's copy of the same logic.
  - A coverage warning: if fewer than half of a sample's requested elements
    are shared with the reference library, a warning is appended rather than
    silently returning a score computed from a handful of elements.
"""

from abc import ABC, abstractmethod
from time import perf_counter

from ..preprocessing import describe, prepare_vectors, resolve_options
from .envelope import Match, RunResult

#: How many nearest references vote on a match's confidence, when an
#: algorithm supports it.
DEFAULT_K = 5

#: Confidence is only computed for this many leading matches — computing it
#: for a full ranking of every reference would multiply the row size of a
#: result set that already runs to hundreds of rows per sample.
DEFAULT_DETAIL_TOP_N = 10

#: Below this fraction of shared elements, a coverage warning is raised.
MIN_COVERAGE_FRACTION = 0.5


def weighted_mean(values, weights=None):
    """
    Average values, honouring per-element weights when there are any.

    weights of None takes the unweighted path rather than multiplying through
    by ones, so a run without weighting produces exactly the arithmetic it
    produced before weighting existed.
    """
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


class SimilarityAlgorithm(ABC):
    """
    One swappable way of scoring an uploaded sample against the reference
    library.

    Adding an algorithm means writing one module and adding one line to
    registry.py. Subclass this directly when the algorithm needs the whole
    reference library in memory at once, for example anything that fits a model
    or builds a projection. Subclass PairwiseSimilarity instead when the
    algorithm scores one reference at a time.

    Every implementation documents its normalisation in the class docstring, as
    required by the contract, because a similarity of 0.8 from one algorithm is
    not comparable to 0.8 from another.
    """

    #: Registry key, and the value clients send as similarity_method.
    id: str = ""
    #: Semantic version. Bump it whenever the arithmetic changes, so stored
    #: runs stay interpretable.
    version: str = "0.0.0"
    #: Subset of {"evidence", "per_sample", "projection"}. Declares which
    #: optional envelope blocks this algorithm fills in.
    capabilities: frozenset = frozenset()

    def raw_scores(self, prepared):
        """
        Return algorithm-native metrics to sit beside the normalised score.

        Distances, correlation coefficients, and probabilities keep their own
        names here. Empty by default, so an algorithm that has nothing extra to
        say does not have to implement it.
        """
        return {}

    def evidence(self, prepared):
        """
        Return (supporting, conflicting) lists of Evidence for one comparison.

        Empty by default. An algorithm that fills this in should declare the
        "evidence" capability so consumers know to look for it.
        """
        return [], []

    def confidence(self, deposit_id, nearest_deposits):
        """
        Judge one match against the deposits of the nearest references.

        None by default, meaning the algorithm offers no calibrated opinion.
        Mix in algorithms.confidence.VotedConfidenceMixin to opt in without
        writing this method yourself.
        """
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
    """
    Base for algorithms that score one sample against one reference at a time.

    Streaming is the reason this exists. The reference library is walked in
    batches so it never has to be held in memory all at once, which is a hard
    constraint on the current hosting. compare() is therefore written as a loop
    over score_pair() rather than as a matrix operation, and the API can call
    score_pair() directly from inside its existing batch loop.

    A subclass implements score_vectors() only. Alignment, preprocessing, the
    empty-overlap case, and confidence (for algorithms that opt in) are handled
    here so that each algorithm is just its own arithmetic.
    """

    def score_pair(
        self,
        input_values,
        reference_values,
        common_elements,
        preprocessing=None,
        imputed_elements=None,
    ):
        """
        Score one tested sample against one reference sample.

        preprocessing accepts either a raw request block or an options dict
        already through resolve_options(), so a caller that resolved once
        outside a loop does not pay for it on every reference.
        """
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
        """
        Return a similarity within [0, 1] for a PreparedVectors instance.

        prepared carries the aligned vectors, the element symbols in matching
        order, the imputed mask, and per-element weights. Both vectors are
        non-empty and the same length. Taking one object rather than several
        positional arguments means a later field can be added without breaking
        algorithms already written against this interface.
        """

    def compare(self, samples, references, config=None):
        """
        Rank every reference against each sample and return a v1.0 RunResult.

        The API calls this once per analysed sample, so samples usually holds a
        single entry. matches therefore ranks the first sample, while
        sample_results carries the top matches for all of them. Whether matches
        should instead be aggregated to deposit level is an open decision, and
        nothing here forecloses it.
        """
        config = config or {}
        top_n = int(config.get("top_n", 200))
        confidence_k = int(config.get("k", DEFAULT_K))
        detail_top_n = int(config.get("detail_top_n", DEFAULT_DETAIL_TOP_N))
        # Resolved once here rather than per reference, since validating the
        # policy names on every one of a thousand comparisons is wasted work.
        options = resolve_options(
            config.get("preprocessing"),
            config.get("selected_elements"),
        )

        # Materialised once so a generator of references can be reused across
        # samples without silently ranking nothing on the second pass.
        reference_list = list(references)
        started = perf_counter()

        rankings = [
            self._rank_one_sample(sample, reference_list, options, top_n)
            for sample in samples
        ]

        # Confidence for algorithms that opt in via VotedConfidenceMixin.
        # self.confidence() returns None for everything else, so this loop is
        # a no-op cost-wise for algorithms that don't implement it.
        if rankings and rankings[0]:
            nearest_deposits = [
                match.deposit_id or match.deposit_name
                for match in rankings[0][:confidence_k]
            ]
            for match in rankings[0][:detail_top_n]:
                match.confidence = self.confidence(
                    match.deposit_id or match.deposit_name,
                    nearest_deposits,
                )

        runtime_ms = (perf_counter() - started) * 1000

        # The element suite that was actually compared, not the one requested.
        # A user can select twenty elements and share only six with a reference.
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

        if samples:
            requested = set(samples[0].get("values") or {})
            if requested and len(elements_used) < len(requested) * MIN_COVERAGE_FRACTION:
                warnings.append(
                    f"Only {len(elements_used)}/{len(requested)} of the sample's "
                    "elements are shared with the reference library; similarity "
                    "scores may be unreliable. Check element naming conventions "
                    "(e.g. 'Au' vs 'Au_ppm')."
                )

        return RunResult(
            algorithm_id=self.id,
            algorithm_version=self.version,
            algorithm_params={"top_n": top_n, "k": confidence_k},
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
