"""Tests for benchmarks, projections and whole-library algorithms."""

from django.test import SimpleTestCase, TestCase

from ..algorithms import get_algorithm
from ..algorithms.base import SimilarityAlgorithm
from ..algorithms.envelope import Match, RunResult
from ..api_views import FullAnalysisListCreateView
from ..benchmarks import (
    LEAVE_ONE_DEPOSIT_OUT,
    LEAVE_ONE_SAMPLE_OUT,
    load_signatures,
    majority_class_baseline,
    run_benchmark,
)
from ..benchmarks.harness import _library_for
from ..models import (
    Element,
    FullAnalysis,
    FullAnalysisMatch,
    ReferenceDeposit,
    ReferenceSample,
    ReferenceSampleMeasurement,
)
from ..projections import fit_pca, fit_tsne, project_points


class ProtocolTests(SimpleTestCase):
    """Check which references each benchmark protocol can use."""

    def signatures(self):
        return [
            {"id": 1, "deposit_pk": 10, "deposit_class": "VHMS"},
            {"id": 2, "deposit_pk": 10, "deposit_class": "VHMS"},
            {"id": 3, "deposit_pk": 20, "deposit_class": "VHMS"},
            {"id": 4, "deposit_pk": 30, "deposit_class": "IOCG"},
        ]

    def test_leave_one_sample_out_hides_only_the_query(self):
        library = _library_for(
            self.signatures()[0], self.signatures(), LEAVE_ONE_SAMPLE_OUT
        )

        # Sample-out keeps another sample from the same deposit.
        self.assertEqual([s["id"] for s in library], [2, 3, 4])

    def test_leave_one_deposit_out_hides_every_sister_sample(self):
        library = _library_for(
            self.signatures()[0], self.signatures(), LEAVE_ONE_DEPOSIT_OUT
        )

        self.assertEqual([s["id"] for s in library], [3, 4])

    def test_an_unknown_protocol_is_rejected(self):
        with self.assertRaises(ValueError):
            run_benchmark(get_algorithm(None), [], protocol="leave_one_vibe_out")

    def test_the_majority_baseline_is_the_commonest_class_share(self):
        self.assertAlmostEqual(
            majority_class_baseline(self.signatures()), 0.75, places=12
        )

    def test_the_baseline_of_an_empty_library_is_zero(self):
        self.assertEqual(majority_class_baseline([]), 0.0)


class HarnessTests(TestCase):
    """Run benchmarks against a small library with known classes."""

    def setUp(self):
        self.elements = {
            symbol: Element.objects.create(symbol=symbol)
            for symbol in ("Cu", "Zn", "Au")
        }
        # Two classes, two deposits each, two samples per deposit. Samples
        # within a class are chemically close and across classes are far.
        self.build("VHMS", "V1", {"Cu": 100.0, "Zn": 10.0, "Au": 1.0})
        self.build("VHMS", "V2", {"Cu": 110.0, "Zn": 11.0, "Au": 1.1})
        self.build("IOCG", "I1", {"Cu": 1.0, "Zn": 10.0, "Au": 100.0})
        self.build("IOCG", "I2", {"Cu": 1.1, "Zn": 11.0, "Au": 110.0})

    def build(self, deposit_class, code, values):
        deposit = ReferenceDeposit.objects.create(
            name=code, three_char_code=code[:3], deposit_type=deposit_class,
        )
        for suffix, scale in (("a", 1.0), ("b", 1.02)):
            reference_sample = ReferenceSample.objects.create(
                reference_deposit=deposit, sample_code=f"{code}{suffix}",
            )
            for symbol, value in values.items():
                ReferenceSampleMeasurement.objects.create(
                    reference_sample=reference_sample,
                    element=self.elements[symbol],
                    value=value * scale,
                    unit="ppm",
                )

    def test_signatures_load_with_their_class_and_deposit(self):
        signatures = load_signatures()

        self.assertEqual(len(signatures), 8)
        self.assertEqual(
            {s["deposit_class"] for s in signatures}, {"VHMS", "IOCG"}
        )
        self.assertEqual(len({s["deposit_pk"] for s in signatures}), 4)

    def test_a_sample_without_a_class_is_excluded(self):
        unlabelled = ReferenceDeposit.objects.create(
            name="Unknown", three_char_code="UNK", deposit_type="",
        )
        reference_sample = ReferenceSample.objects.create(
            reference_deposit=unlabelled, sample_code="U1",
        )
        ReferenceSampleMeasurement.objects.create(
            reference_sample=reference_sample,
            element=self.elements["Cu"], value=50.0, unit="ppm",
        )

        self.assertEqual(len(load_signatures()), 8)

    def test_both_protocols_retrieve_the_right_class_on_separable_data(self):
        signatures = load_signatures()

        for protocol in (LEAVE_ONE_SAMPLE_OUT, LEAVE_ONE_DEPOSIT_OUT):
            with self.subTest(protocol=protocol):
                result = run_benchmark(
                    get_algorithm("knn_aitchison"), signatures, protocol=protocol
                )

                self.assertEqual(result.n_queries, 8)
                self.assertAlmostEqual(result.top_1, 1.0, places=12)
                self.assertGreater(result.lift_over_baseline, 0.0)

    def test_the_harness_scores_every_registered_algorithm(self):
        signatures = load_signatures()

        for algorithm_id in ("log_difference_similarity", "correlation", "distance"):
            with self.subTest(algorithm=algorithm_id):
                result = run_benchmark(get_algorithm(algorithm_id), signatures)

                self.assertEqual(result.algorithm_id, algorithm_id)
                self.assertEqual(result.n_queries, 8)
                self.assertGreaterEqual(result.top_1, 0.0)
                self.assertLessEqual(result.top_1, 1.0)

    def test_metrics_stay_within_their_ranges_and_relate_correctly(self):
        result = run_benchmark(get_algorithm("knn_aitchison"), load_signatures())

        self.assertLessEqual(result.top_1, result.top_5)
        self.assertLessEqual(result.mean_reciprocal_rank, 1.0)
        self.assertGreaterEqual(result.mean_reciprocal_rank, result.top_1 * 0.99)
        self.assertGreater(result.runtime_ms, 0.0)
        self.assertAlmostEqual(
            result.ms_per_query, result.runtime_ms / result.n_queries, places=9
        )

    def test_the_query_sample_is_seeded_so_algorithms_are_compared_fairly(self):
        signatures = load_signatures()

        first = run_benchmark(get_algorithm("distance"), signatures, max_queries=3)
        second = run_benchmark(get_algorithm("distance"), signatures, max_queries=3)

        self.assertEqual(first.n_queries, second.n_queries)
        self.assertEqual(first.top_1, second.top_1)

    def test_the_per_class_breakdown_accounts_for_every_query(self):
        result = run_benchmark(get_algorithm("knn_aitchison"), load_signatures())

        self.assertEqual(
            sum(bucket["n"] for bucket in result.per_class.values()),
            result.n_queries,
        )

    def test_the_result_serialises_for_reporting(self):
        payload = run_benchmark(
            get_algorithm("knn_aitchison"), load_signatures()
        ).to_dict()

        self.assertEqual(payload["algorithm"]["id"], "knn_aitchison")
        self.assertIn("top_1", payload)
        self.assertIn("lift_over_baseline", payload)


class PCATests(SimpleTestCase):
    """Check fitting and reusing the PCA projection."""

    def rows(self):
        # Two groups differing in which element dominates, so a first component
        # separating them must exist.
        return (
            [{"Cu": 100.0 * s, "Zn": 10.0, "Au": 1.0} for s in (1.0, 1.1, 0.9, 1.05)]
            + [{"Cu": 1.0, "Zn": 10.0, "Au": 100.0 * s} for s in (1.0, 1.1, 0.9, 1.05)]
        )

    def test_a_model_fits_and_reports_its_axes(self):
        model = fit_pca(self.rows())

        self.assertEqual(model.symbols, ["Au", "Cu", "Zn"])
        self.assertEqual(len(model.components), 2)
        self.assertEqual(len(model.explained_variance_ratio), 2)
        self.assertGreater(model.explained_variance_ratio[0], 0.5)

    def test_variance_ratios_are_ordered_and_bounded(self):
        ratios = fit_pca(self.rows()).explained_variance_ratio

        self.assertGreaterEqual(ratios[0], ratios[1])
        self.assertLessEqual(sum(ratios), 1.0 + 1e-9)

    def test_the_two_groups_land_apart_on_the_first_component(self):
        rows = self.rows()
        model = fit_pca(rows)
        positions = [model.project(row)[0] for row in rows]

        first_centre = sum(positions[:4]) / 4
        second_centre = sum(positions[4:]) / 4
        self.assertGreater(abs(first_centre - second_centre), 2.0)

    def test_component_signs_have_a_stable_orientation(self):
        model = fit_pca(self.rows())

        for component in model.components:
            largest = max(component, key=abs)
            self.assertGreaterEqual(largest, 0.0)

    def test_a_new_sample_is_placed_without_refitting(self):
        model = fit_pca(self.rows())

        placed = model.project({"Cu": 105.0, "Zn": 10.0, "Au": 1.0})
        group = [model.project(row)[0] for row in self.rows()[:4]]

        self.assertIsNotNone(placed)
        self.assertGreaterEqual(placed[0], min(group) - 1.0)
        self.assertLessEqual(placed[0], max(group) + 1.0)

    def test_dilution_does_not_move_a_point(self):
        model = fit_pca(self.rows())

        original = model.project({"Cu": 100.0, "Zn": 10.0, "Au": 1.0})
        diluted = model.project({"Cu": 10.0, "Zn": 1.0, "Au": 0.1})

        self.assertAlmostEqual(original[0], diluted[0], places=9)
        self.assertAlmostEqual(original[1], diluted[1], places=9)

    def test_too_few_rows_yields_no_model_rather_than_noise(self):
        self.assertIsNone(fit_pca([{"Cu": 1.0, "Zn": 2.0}]))
        self.assertIsNone(fit_pca([]))

    def test_a_sample_sharing_too_little_chemistry_is_not_placed(self):
        model = fit_pca(self.rows())

        self.assertIsNone(model.project({"Pb": 5.0}))

    def test_unplaceable_points_are_dropped_rather_than_sent_as_nulls(self):
        model = fit_pca(self.rows())

        points = project_points(model, [
            ("S001", "sample", {"Cu": 100.0, "Zn": 10.0, "Au": 1.0}),
            ("S002", "sample", {"Pb": 5.0}),
        ])

        self.assertEqual([point["id"] for point in points], ["S001"])
        self.assertEqual(set(points[0]), {"id", "x", "y", "kind"})

    def test_projecting_with_no_model_is_safe(self):
        self.assertEqual(project_points(None, [("S001", "sample", {})]), [])


class TSNETests(SimpleTestCase):
    """Check basic t-SNE output and repeatability."""

    def test_the_embedding_has_one_finite_point_per_input(self):
        vectors = [
            [float(i % 4), float(i // 4), float(i % 3)]
            for i in range(20)
        ]

        embedding = fit_tsne(vectors, perplexity=5.0, iterations=30)

        self.assertEqual(len(embedding), 20)
        for x, y in embedding:
            self.assertEqual(x, x)  # not NaN
            self.assertEqual(y, y)
            self.assertLess(abs(x), 1e6)
            self.assertLess(abs(y), 1e6)

    def test_the_embedding_is_reproducible_for_an_unchanged_library(self):
        vectors = [[float(i), float(i % 5)] for i in range(12)]

        first = fit_tsne(vectors, perplexity=3.0, iterations=20)
        second = fit_tsne(vectors, perplexity=3.0, iterations=20)

        self.assertEqual(first, second)

    def test_a_degenerate_input_does_not_raise(self):
        self.assertEqual(len(fit_tsne([[1.0, 2.0]])), 1)
        self.assertEqual(fit_tsne([]), [])


class MatrixAlgorithm(SimilarityAlgorithm):
    """Test algorithm that scores references against the library centroid."""

    id = "test_matrix_algorithm"
    version = "0.1.0"
    capabilities = frozenset()

    def compare(self, samples, references, config=None):
        references = list(references)
        symbols = sorted({s for r in references for s in (r.get("values") or {})})
        centroid = {
            symbol: sum(
                (r.get("values") or {}).get(symbol, 0.0) for r in references
            ) / max(1, len(references))
            for symbol in symbols
        }

        scored = sorted(
            (
                (
                    1 / (1 + sum(
                        abs((r.get("values") or {}).get(s, 0.0) - centroid[s])
                        for s in symbols
                    )),
                    r,
                )
                for r in references
            ),
            key=lambda item: (-item[0], -(item[1].get("id") or 0)),
        )

        return RunResult(
            algorithm_id=self.id,
            algorithm_version=self.version,
            matches=[
                Match(rank=rank, similarity=score,
                      reference_sample_id=reference.get("id"),
                      deposit_id=reference.get("deposit_id", ""))
                for rank, (score, reference) in enumerate(scored, start=1)
            ],
            runtime_ms=0.0,
        )


class MatrixExecutionPathTests(TestCase):
    """Check the API path for algorithms that need the whole library."""

    def setUp(self):
        self.view = FullAnalysisListCreateView()
        self.elements = {
            symbol: Element.objects.create(symbol=symbol)
            for symbol in ("Cu", "Zn", "Au")
        }
        deposit = ReferenceDeposit.objects.create(
            name="Alpha", three_char_code="ALP", deposit_type="VHMS",
        )
        for code, values in (
            ("R1", {"Cu": 100.0, "Zn": 10.0, "Au": 1.0}),
            ("R2", {"Cu": 1000.0, "Zn": 10.0, "Au": 1.0}),
            ("R3", {"Cu": 10.0, "Zn": 100.0, "Au": 1.0}),
        ):
            reference_sample = ReferenceSample.objects.create(
                reference_deposit=deposit, sample_code=code,
            )
            for symbol, value in values.items():
                ReferenceSampleMeasurement.objects.create(
                    reference_sample=reference_sample,
                    element=self.elements[symbol], value=value, unit="ppm",
                )

        self.full_analysis = FullAnalysis.objects.create(
            name="Matrix analysis", uploaded_sample_code="S001", parameters={},
        )
        self.measurements = [
            {"element_symbol": symbol, "value": value, "unit": "ppm",
             "below_detection_limit": False, "detection_limit": None}
            for symbol, value in (("Cu", 100.0), ("Zn", 10.0), ("Au", 1.0))
        ]

    def test_a_non_pairwise_algorithm_produces_persisted_matches(self):
        from ..algorithms import ALGORITHMS

        ALGORITHMS[MatrixAlgorithm.id] = MatrixAlgorithm
        try:
            self.view.create_ranked_matches(
                self.full_analysis, self.measurements, 0, 10, 250,
                MatrixAlgorithm.id, {},
            )
        finally:
            ALGORITHMS.pop(MatrixAlgorithm.id, None)

        matches = list(
            FullAnalysisMatch.objects
            .filter(full_analysis=self.full_analysis)
            .order_by("rank")
        )

        self.assertEqual(len(matches), 3)
        self.assertEqual([m.rank for m in matches], [1, 2, 3])
        # Descending similarity, and every score a real float.
        scores = [m.similarity_score for m in matches]
        self.assertEqual(scores, sorted(scores, reverse=True))
        for score in scores:
            self.assertIsInstance(score, float)
