"""Compare similarity algorithms against the reference library."""

import json

from django.core.management.base import BaseCommand, CommandError

from ...algorithms import ALGORITHMS, get_algorithm
from ...benchmarks import (
    LEAVE_ONE_DEPOSIT_OUT,
    PROTOCOLS,
    load_signatures,
    run_benchmark,
)


class Command(BaseCommand):
    help = (
        "Score similarity algorithms with leave-one-out retrieval and report "
        "which performs best."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--algorithms",
            nargs="*",
            help="Algorithm ids to compare. Defaults to every registered one.",
        )
        parser.add_argument(
            "--protocol",
            action="append",
            choices=PROTOCOLS,
            help=(
                "Evaluation protocol; repeatable. Defaults to leave-one-deposit-out, "
                "which is the honest one."
            ),
        )
        parser.add_argument(
            "--max-queries",
            type=int,
            default=400,
            help=(
                "Sample this many queries. 0 evaluates all. Below roughly 300 "
                "the gap between the two leading algorithms is inside the "
                "sampling noise, so a small run can rank them the wrong way."
            ),
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=7,
            help=(
                "Seed for the query sample. Every algorithm sees the same "
                "queries, so a comparison is never also a comparison of two "
                "different random subsets. Vary it to check a result is stable."
            ),
        )
        parser.add_argument(
            "--handle-missing",
            default="half_dl",
            help="Censored-data policy applied to every algorithm equally.",
        )
        parser.add_argument(
            "--per-class",
            action="store_true",
            help="Also print a per-class breakdown for the best algorithm.",
        )
        parser.add_argument(
            "--json",
            dest="as_json",
            action="store_true",
            help="Emit machine-readable results instead of a table.",
        )

    def handle(self, *args, **options):
        algorithm_ids = options["algorithms"] or list(ALGORITHMS)
        unknown = [name for name in algorithm_ids if name not in ALGORITHMS]
        if unknown:
            raise CommandError(
                f"Unknown algorithm(s): {', '.join(unknown)}. "
                f"Available: {', '.join(ALGORITHMS)}"
            )

        protocols = options["protocol"] or [LEAVE_ONE_DEPOSIT_OUT]
        max_queries = options["max_queries"] or None
        preprocessing = {"handle_missing": options["handle_missing"]}

        self.stdout.write("Loading reference library...")
        signatures = load_signatures(preprocessing)
        if not signatures:
            raise CommandError(
                "No reference samples with a deposit class were found. "
                "Import the reference library first."
            )

        deposits = len({s["deposit_pk"] for s in signatures})
        classes = len({s["deposit_class"] for s in signatures})
        self.stdout.write(
            f"  {len(signatures)} samples, {deposits} deposits, {classes} classes"
        )

        results = []
        for protocol in protocols:
            for algorithm_id in algorithm_ids:
                self.stdout.write(f"  scoring {algorithm_id} ({protocol})...")
                results.append(run_benchmark(
                    get_algorithm(algorithm_id),
                    signatures,
                    protocol=protocol,
                    max_queries=max_queries,
                    seed=options["seed"],
                    preprocessing=preprocessing,
                ))

        if options["as_json"]:
            self.stdout.write(json.dumps(
                [result.to_dict() for result in results],
                indent=2,
            ))
            return

        self.report(results, per_class=options["per_class"])

    def report(self, results, per_class=False):
        baseline = results[0].majority_baseline

        self.stdout.write("")
        self.stdout.write(
            f"{'algorithm':<28}{'protocol':<10}{'top-1':>8}{'top-5':>8}"
            f"{'MRR':>8}{'lift':>8}{'ms/query':>10}"
        )
        self.stdout.write("-" * 80)

        for result in results:
            self.stdout.write(
                f"{result.algorithm_id:<28}{result.protocol:<10}"
                f"{result.top_1:>7.1%}{result.top_5:>8.1%}"
                f"{result.mean_reciprocal_rank:>8.3f}"
                f"{result.lift_over_baseline:>+8.1%}{result.ms_per_query:>10.1f}"
            )

        self.stdout.write("-" * 80)
        self.stdout.write(f"majority-class baseline: {baseline:.1%}")

        best = max(results, key=lambda result: (result.top_1, result.top_5))
        self.stdout.write(self.style.SUCCESS(
            f"best: {best.algorithm_id} ({best.protocol}) "
            f"top-1 {best.top_1:.1%}, top-5 {best.top_5:.1%}"
        ))

        # A ranking decided by fewer queries than the gap between the top two
        # can support is not a ranking. Say so rather than letting the table
        # imply more precision than it has.
        contenders = sorted(
            (r for r in results if r.protocol == best.protocol),
            key=lambda result: -result.top_1,
        )
        if len(contenders) >= 2:
            leader = contenders[0]
            gap = leader.top_1 - contenders[1].top_1
            # Standard error of the leader's own accuracy. A rough yardstick,
            # not a significance test: both algorithms answer the same queries,
            # so the error on the difference is smaller than this. It is here to
            # prompt a re-run when a gap is obviously too small to trust.
            standard_error = (
                leader.top_1 * (1 - leader.top_1) / max(1, leader.n_queries)
            ) ** 0.5
            if gap < standard_error:
                self.stdout.write(self.style.WARNING(
                    f"  the top two differ by {gap:.1%}, comparable to the "
                    f"+/-{standard_error:.1%} standard error at "
                    f"{leader.n_queries} queries. Re-run with more queries or "
                    f"another --seed before calling one better."
                ))

        # A result near the baseline means the algorithm is not working, and a
        # very high deposit-out result usually means the protocol leaked.
        for result in results:
            if result.top_1 <= baseline:
                self.stdout.write(self.style.ERROR(
                    f"  {result.algorithm_id} ({result.protocol}) does not beat "
                    f"the baseline. Treat it as broken."
                ))
            elif result.protocol == LEAVE_ONE_DEPOSIT_OUT and result.top_1 > 0.69:
                self.stdout.write(self.style.WARNING(
                    f"  {result.algorithm_id} scores {result.top_1:.1%} on the "
                    f"deposit-out protocol. Check for leakage."
                ))

        if per_class:
            self.report_per_class(best)

    def report_per_class(self, result):
        self.stdout.write("")
        self.stdout.write(f"per-class breakdown for {result.algorithm_id}:")
        self.stdout.write(f"{'class':<38}{'n':>6}{'top-1':>9}{'top-5':>9}")
        self.stdout.write("-" * 62)

        ordered = sorted(
            result.per_class.items(),
            key=lambda item: -item[1]["n"],
        )
        for label, bucket in ordered:
            self.stdout.write(
                f"{label[:37]:<38}{bucket['n']:>6}"
                f"{bucket['top_1_rate']:>8.0%}{bucket['top_5_rate']:>9.0%}"
            )
