"""Leave-one-out evaluation for registered similarity algorithms.

Sample-out removes the query sample. Deposit-out removes every reference from
the query deposit and gives a stricter measure of deposit-class retrieval.
"""

from collections import Counter
from dataclasses import dataclass, field
from time import perf_counter
import random

from ..preprocessing import extract_values, resolve_options

LEAVE_ONE_SAMPLE_OUT = "sample"
LEAVE_ONE_DEPOSIT_OUT = "deposit"
PROTOCOLS = (LEAVE_ONE_SAMPLE_OUT, LEAVE_ONE_DEPOSIT_OUT)

# Field used as the target class during evaluation.
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
        """Return the top-1 improvement over the majority baseline."""
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

    The model import stays inside this function so Django can finish setup
    before the query is built.
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
            # Unlabelled samples cannot be scored as retrieval hits.
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

    This gives the accuracy of always selecting the most common class.
    """
    if not signatures:
        return 0.0

    counts = Counter(signature["deposit_class"] for signature in signatures)
    return counts.most_common(1)[0][1] / len(signatures)


def _library_for(query, signatures, protocol):
    """Return the references a query is allowed to be matched against."""
    if protocol == LEAVE_ONE_DEPOSIT_OUT:
        # Withhold every sample belonging to the query deposit.
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
        # A fixed seed gives each algorithm the same query subset.
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

        # All registered algorithms use the same compare interface.
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
