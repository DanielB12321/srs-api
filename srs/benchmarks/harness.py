"""
Leave-one-out evaluation of any registered similarity algorithm.

The harness reads nothing but the ranked matches an algorithm returns, so it
scores a compositional algorithm, a correlation algorithm, and anything added
later through exactly the same code path. Nothing here knows how a similarity
was arrived at.

Two protocols, and the difference between them matters more than any single
number:

* Leave one sample out hides only the query sample. Other samples from the same
  deposit stay in the library, and they are usually near-duplicates of it, so
  the score this produces is optimistic.
* Leave one deposit out hides every sample from the query's deposit. The
  algorithm has to recognise the deposit type from a different deposit
  entirely, which is the question a user is actually asking.

Report the deposit-out figure. The sample-out figure is useful only as the
gap between them, which tells you how much of the apparent accuracy was
sister samples.
"""

from collections import Counter
from dataclasses import dataclass, field
from time import perf_counter
import random

from ..preprocessing import extract_values, resolve_options

LEAVE_ONE_SAMPLE_OUT = "sample"
LEAVE_ONE_DEPOSIT_OUT = "deposit"
PROTOCOLS = (LEAVE_ONE_SAMPLE_OUT, LEAVE_ONE_DEPOSIT_OUT)

#: Which field on the deposit carries the class label being retrieved.
#: deposit_type is the only populated label with a join path to a deposit;
#: DepositClassification exists but nothing links it to ReferenceDeposit.
DEFAULT_CLASS_FIELD = "deposit_type"


@dataclass
class BenchmarkResult:
    """One algorithm measured under one protocol."""

    algorithm_id: str
    algorithm_version: str
    protocol: str
    n_queries: int
    top_1: float
    top_5: float
    mean_reciprocal_rank: float
    majority_baseline: float
    runtime_ms: float
    ms_per_query: float
    per_class: dict = field(default_factory=dict)

    @property
    def lift_over_baseline(self):
        """How much better than always guessing the commonest class."""
        return self.top_1 - self.majority_baseline

    def to_dict(self):
        return {
            "algorithm": {
                "id": self.algorithm_id,
                "version": self.algorithm_version,
            },
            "protocol": self.protocol,
            "n_queries": self.n_queries,
            "top_1": self.top_1,
            "top_5": self.top_5,
            "mean_reciprocal_rank": self.mean_reciprocal_rank,
            "majority_baseline": self.majority_baseline,
            "lift_over_baseline": self.lift_over_baseline,
            "runtime_ms": self.runtime_ms,
            "ms_per_query": self.ms_per_query,
            "per_class": self.per_class,
        }


def load_signatures(preprocessing=None, class_field=DEFAULT_CLASS_FIELD):
    """
    Read the reference library into the plain dicts an algorithm expects.

    Imported here rather than at module scope so the harness can be imported
    without Django models being ready, which keeps it usable from a plain
    script as well as a management command.
    """
    from ..models import ReferenceSample

    options = resolve_options(preprocessing)
    signatures = []

    reference_samples = (
        ReferenceSample.objects
        .select_related("reference_deposit")
        .prefetch_related("measurements__element")
        .order_by("id")
    )

    for reference_sample in reference_samples:
        deposit = reference_sample.reference_deposit
        if deposit is None:
            continue

        label = (getattr(deposit, class_field, "") or "").strip()
        if not label:
            # A sample with no class cannot be scored as a retrieval hit, and
            # leaving it in the library would let it displace one that can.
            continue

        values, imputed = extract_values(reference_sample.measurements.all(), options)
        if not values:
            continue

        signatures.append({
            "id": reference_sample.id,
            "values": values,
            "imputed": imputed,
            "deposit_pk": deposit.id,
            "deposit_id": deposit.three_char_code or deposit.name,
            "deposit_name": deposit.name,
            "deposit_class": label,
        })

    return signatures


def majority_class_baseline(signatures):
    """
    The score from always guessing the commonest class.

    An algorithm that cannot beat this has learned nothing, whatever its
    absolute accuracy looks like.
    """
    if not signatures:
        return 0.0

    counts = Counter(signature["deposit_class"] for signature in signatures)
    return counts.most_common(1)[0][1] / len(signatures)


def _library_for(query, signatures, protocol):
    """Return the references a query is allowed to be matched against."""
    if protocol == LEAVE_ONE_DEPOSIT_OUT:
        # Every sample from the query's deposit is withheld, not just the query
        # itself. Sister samples from one deposit are near-duplicates, and
        # leaving them in is the single easiest way to overstate accuracy.
        return [
            signature
            for signature in signatures
            if signature["deposit_pk"] != query["deposit_pk"]
        ]

    return [
        signature
        for signature in signatures
        if signature["id"] != query["id"]
    ]


def run_benchmark(
    algorithm,
    signatures,
    protocol=LEAVE_ONE_DEPOSIT_OUT,
    max_queries=None,
    seed=7,
    top_n=5,
    preprocessing=None,
    progress=None,
):
    """
    Score one algorithm and return a BenchmarkResult.

    max_queries samples the query set for a quick run; the sample is drawn with
    a fixed seed so two algorithms are always compared on identical queries.
    Leaving it None evaluates every sample.
    """
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {PROTOCOLS}, got {protocol!r}")

    queries = list(signatures)
    if max_queries is not None and max_queries < len(queries):
        # Seeded so a comparison between algorithms is not also a comparison
        # between two different random query sets.
        queries = random.Random(seed).sample(queries, max_queries)

    baseline = majority_class_baseline(signatures)
    config = {"top_n": top_n, "preprocessing": preprocessing or {}}

    hits_at_1 = 0
    hits_at_5 = 0
    reciprocal_ranks = 0.0
    scored_queries = 0
    per_class = {}
    started = perf_counter()

    for position, query in enumerate(queries):
        library = _library_for(query, signatures, protocol)
        if not library:
            continue

        # compare() is the only interface used, which is what lets this score
        # an algorithm it has never heard of.
        result = algorithm.compare(
            [{"sample_code": str(query["id"]), "values": query["values"],
              "imputed": query["imputed"]}],
            library,
            config,
        )
        matches = result.matches
        if not matches:
            continue

        scored_queries += 1
        truth = query["deposit_class"]
        by_id = {signature["id"]: signature for signature in library}
        retrieved = [
            by_id[match.reference_sample_id]["deposit_class"]
            for match in matches
            if match.reference_sample_id in by_id
        ]

        bucket = per_class.setdefault(truth, {"n": 0, "top_1": 0, "top_5": 0})
        bucket["n"] += 1

        if retrieved[:1] == [truth]:
            hits_at_1 += 1
            bucket["top_1"] += 1
        if truth in retrieved[:5]:
            hits_at_5 += 1
            bucket["top_5"] += 1

        for rank, label in enumerate(retrieved, start=1):
            if label == truth:
                reciprocal_ranks += 1 / rank
                break

        if progress is not None:
            progress(position + 1, len(queries))

    elapsed_ms = (perf_counter() - started) * 1000
    divisor = scored_queries or 1

    for bucket in per_class.values():
        bucket["top_1_rate"] = bucket["top_1"] / bucket["n"]
        bucket["top_5_rate"] = bucket["top_5"] / bucket["n"]

    return BenchmarkResult(
        algorithm_id=algorithm.id,
        algorithm_version=algorithm.version,
        protocol=protocol,
        n_queries=scored_queries,
        top_1=hits_at_1 / divisor,
        top_5=hits_at_5 / divisor,
        mean_reciprocal_rank=reciprocal_ranks / divisor,
        majority_baseline=baseline,
        runtime_ms=elapsed_ms,
        ms_per_query=elapsed_ms / divisor,
        per_class=per_class,
    )
