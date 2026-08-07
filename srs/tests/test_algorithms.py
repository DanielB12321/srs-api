"""
Contract tests for the pluggable algorithm layer.

The characterisation suite proves the extraction did not change any score.
These prove the layer around it holds up: every registered algorithm satisfies
the envelope contract, the registry resolves the way callers expect, and the
view's delegation reaches the same code the classes expose directly.

The envelope checks loop over the registry rather than naming algorithms, so an
algorithm added later is held to the same contract without anyone remembering
to write a test for it.
"""

from django.test import SimpleTestCase, TestCase, override_settings

from ..algorithms import (
    ALGORITHMS,
    FALLBACK_ALGORITHM_ID,
    SCHEMA_VERSION,
    SimilarityAlgorithm,
    available_algorithms,
    default_algorithm_id,
    get_algorithm,
    validate_envelope,
)
from ..algorithms.envelope import Evidence, Match, RunResult
from ..api_views import FullAnalysisListCreateView
from ..models import (
    Element,
    FullAnalysis,
    FullAnalysisMatch,
    ReferenceDeposit,
    ReferenceSample,
    ReferenceSampleMeasurement,
)
from ..preprocessing import PIPELINE_VERSION
from .test_characterisation import (
    COMMON_ELEMENTS,
    EXPECTED_SCORES,
    INPUT_VALUES,
    PREPROCESSING_STATES,
    REFERENCE_VALUES,
)

SAMPLES = [{
    "sample_code": "S001",
    "latitude": -35.06,
    "longitude": 149.57,
    "values": INPUT_VALUES,
}]

REFERENCES = [
    {
        "id": index,
        "values": values,
        "deposit_id": "WLN",
        "deposit_name": "Woodlawn",
        "deposit_class": "VHMS",
    }
    for index, values in enumerate(REFERENCE_VALUES.values(), start=1)
]


class RegistryTests(SimpleTestCase):
    """Resolution rules that callers and the API contract depend on."""

    def test_every_registered_algorithm_declares_its_identity(self):
        for algorithm_id, algorithm_class in ALGORITHMS.items():
            with self.subTest(algorithm=algorithm_id):
                self.assertTrue(issubclass(algorithm_class, SimilarityAlgorithm))
                # The registry key and the class attribute must agree, or
                # provenance recorded against a run points at the wrong thing.
                self.assertEqual(algorithm_class.id, algorithm_id)
                self.assertTrue(algorithm_class.version)
                self.assertTrue(algorithm_class.__doc__, "needs a docstring")

    def test_known_ids_resolve_to_their_own_algorithm(self):
        for algorithm_id in ALGORITHMS:
            with self.subTest(algorithm=algorithm_id):
                self.assertEqual(get_algorithm(algorithm_id).id, algorithm_id)

    def test_unknown_and_missing_ids_resolve_to_the_default(self):
        for requested in ("not_a_real_method", "", None):
            with self.subTest(requested=requested):
                self.assertEqual(get_algorithm(requested).id, default_algorithm_id())

    @override_settings(SRS_DEFAULT_ALGORITHM="distance")
    def test_the_default_is_configurable_without_touching_algorithm_code(self):
        self.assertEqual(default_algorithm_id(), "distance")
        self.assertEqual(get_algorithm(None).id, "distance")
        # An explicit request still wins over the deployment default.
        self.assertEqual(get_algorithm("correlation").id, "correlation")

    @override_settings(SRS_DEFAULT_ALGORITHM="a_typo_in_the_environment")
    def test_a_misconfigured_default_does_not_break_the_service(self):
        self.assertEqual(default_algorithm_id(), FALLBACK_ALGORITHM_ID)
        self.assertEqual(get_algorithm(None).id, FALLBACK_ALGORITHM_ID)

    def test_instances_are_not_shared_between_calls(self):
        """Analyses run on background threads, so state must not be shared."""
        self.assertIsNot(
            get_algorithm("log_difference_similarity"),
            get_algorithm("log_difference_similarity"),
        )

    def test_available_algorithms_describes_the_whole_registry(self):
        described = available_algorithms()

        self.assertEqual(
            {entry["id"] for entry in described},
            set(ALGORITHMS),
        )
        self.assertEqual(
            [entry["id"] for entry in described if entry["is_default"]],
            [default_algorithm_id()],
        )
        for entry in described:
            with self.subTest(algorithm=entry["id"]):
                self.assertTrue(entry["description"])
                self.assertTrue(entry["version"])


class ViewDelegationTests(SimpleTestCase):
    """
    The view must reach the extracted classes, not a copy of the arithmetic.

    This is what stops the two paths drifting apart: if someone later edits an
    algorithm module but the view keeps its own maths, these fail.
    """

    def setUp(self):
        self.view = FullAnalysisListCreateView()

    def test_the_view_and_the_algorithm_classes_agree_exactly(self):
        for method, by_preprocessing in EXPECTED_SCORES.items():
            algorithm = get_algorithm(method)
            for state_name, preprocessing in PREPROCESSING_STATES.items():
                for reference_key, reference_values in REFERENCE_VALUES.items():
                    with self.subTest(
                        method=method,
                        preprocessing=state_name,
                        reference=reference_key,
                    ):
                        self.assertEqual(
                            self.view.calculate_similarity_score(
                                INPUT_VALUES,
                                reference_values,
                                COMMON_ELEMENTS,
                                method,
                                preprocessing,
                            ),
                            algorithm.score_pair(
                                INPUT_VALUES,
                                reference_values,
                                COMMON_ELEMENTS,
                                preprocessing,
                            ),
                        )

    def test_the_view_holds_no_similarity_arithmetic(self):
        """
        Delegation, not a reimplementation.

        A tolerant ceiling rather than an exact line count, so ordinary edits do
        not fail it, but pasting a formula back in would.
        """
        import inspect

        source = inspect.getsource(self.view.calculate_similarity_score)
        body = [
            line.strip()
            for line in source.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

        self.assertNotIn("log10", source)
        self.assertLess(len(body), 25)


class DispatchReachesThePersistedRankingTests(TestCase):
    """
    The Phase 2 deliverable, checked where it actually matters.

    Resolving an algorithm is only useful if the choice survives all the way to
    the stored FullAnalysisMatch rows. These select a method, run the real
    ranking path, and read the scores back out of the database.
    """

    def setUp(self):
        self.view = FullAnalysisListCreateView()
        self.elements = {
            symbol: Element.objects.create(symbol=symbol)
            for symbol in ("Cu", "Zn", "Au")
        }
        deposit = ReferenceDeposit.objects.create(name="Alpha", three_char_code="ALP")
        reference_sample = ReferenceSample.objects.create(
            reference_deposit=deposit,
            sample_code="R_swapped",
        )
        for symbol, value in REFERENCE_VALUES["swapped"].items():
            ReferenceSampleMeasurement.objects.create(
                reference_sample=reference_sample,
                element=self.elements[symbol],
                value=value,
                unit="ppm",
            )

        self.full_analysis = FullAnalysis.objects.create(
            name="Dispatch analysis",
            uploaded_sample_code="S001",
        )
        self.measurements = [
            {
                "element_symbol": symbol,
                "value": value,
                "unit": "ppm",
                "below_detection_limit": False,
                "detection_limit": None,
            }
            for symbol, value in INPUT_VALUES.items()
        ]

    def score_via_ranking_path(self, similarity_method, preprocessing=None):
        self.view.create_ranked_matches(
            self.full_analysis,
            self.measurements,
            0,
            10,
            250,
            similarity_method,
            preprocessing or {},
        )
        return FullAnalysisMatch.objects.get(
            full_analysis=self.full_analysis,
            analysed_sample_index=0,
            rank=1,
        ).similarity_score

    def test_each_method_produces_its_own_score_in_the_database(self):
        """
        The reference is deliberately the 'swapped' one, because that is where
        the four methods disagree most. Identical scores would prove nothing.
        """
        for method, by_preprocessing in EXPECTED_SCORES.items():
            with self.subTest(method=method):
                self.assertAlmostEqual(
                    self.score_via_ranking_path(method),
                    by_preprocessing["no_preprocessing"]["swapped"],
                    places=12,
                )

    def test_preprocessing_reaches_the_algorithm_through_the_ranking_path(self):
        self.assertAlmostEqual(
            self.score_via_ranking_path("distance", {"normalise": True}),
            EXPECTED_SCORES["distance"]["normalise"]["swapped"],
            places=12,
        )

    @override_settings(SRS_DEFAULT_ALGORITHM="association")
    def test_the_configured_default_is_what_runs_when_none_is_named(self):
        """Changing one setting changes which algorithm the API actually uses."""
        self.assertAlmostEqual(
            self.score_via_ranking_path(None),
            EXPECTED_SCORES["association"]["no_preprocessing"]["swapped"],
            places=12,
        )


class EnvelopeContractTests(SimpleTestCase):
    """Rules from the v1.0 envelope specification, applied to every algorithm."""

    def envelope_for(self, algorithm_id):
        return get_algorithm(algorithm_id).compare(
            SAMPLES,
            REFERENCES,
            {"top_n": 10, "preprocessing": {}},
        ).to_envelope()

    def test_every_algorithm_produces_a_valid_envelope(self):
        for algorithm_id in ALGORITHMS:
            with self.subTest(algorithm=algorithm_id):
                self.assertEqual(validate_envelope(self.envelope_for(algorithm_id)), [])

    def test_similarity_is_present_as_a_float_within_range(self):
        """Contract rule 1, the rule that makes algorithms comparable at all."""
        for algorithm_id in ALGORITHMS:
            envelope = self.envelope_for(algorithm_id)
            self.assertTrue(envelope["matches"], f"{algorithm_id} ranked nothing")

            for match in envelope["matches"]:
                with self.subTest(algorithm=algorithm_id, rank=match["rank"]):
                    similarity = match["scores"]["similarity"]
                    self.assertIsInstance(similarity, float)
                    self.assertGreaterEqual(similarity, 0.0)
                    self.assertLessEqual(similarity, 1.0)

    def test_association_integer_score_is_coerced_at_the_envelope_boundary(self):
        """
        The quirk the characterisation suite recorded, handled where it should
        be. The algorithm keeps its original arithmetic; the envelope makes it
        a float.
        """
        envelope = self.envelope_for("association")
        top_match = envelope["matches"][0]

        self.assertEqual(top_match["scores"]["similarity"], 1.0)
        self.assertIsInstance(top_match["scores"]["similarity"], float)

    def test_matches_are_ranked_best_first_and_numbered_from_one(self):
        for algorithm_id in ALGORITHMS:
            with self.subTest(algorithm=algorithm_id):
                matches = self.envelope_for(algorithm_id)["matches"]
                similarities = [match["scores"]["similarity"] for match in matches]

                self.assertEqual(
                    [match["rank"] for match in matches],
                    list(range(1, len(matches) + 1)),
                )
                self.assertEqual(similarities, sorted(similarities, reverse=True))

    def test_the_run_block_carries_enough_to_reproduce_the_run(self):
        """Contract rule 4."""
        envelope = get_algorithm("correlation").compare(
            SAMPLES,
            REFERENCES,
            {
                "top_n": 3,
                "preprocessing": {"normalise": True},
                "reference_library_version": "osnaca-2026-05",
                "dataset_id": 12,
            },
        ).to_envelope()
        run = envelope["run"]

        self.assertEqual(envelope["schema_version"], SCHEMA_VERSION)
        self.assertEqual(run["algorithm"]["id"], "correlation")
        self.assertEqual(run["algorithm"]["version"], "1.0.0")
        self.assertEqual(run["algorithm"]["params"]["top_n"], 3)
        # The preprocessing block records the resolved configuration and the
        # element suite that was actually compared, not the raw request.
        self.assertEqual(run["preprocessing"]["pipeline_version"], PIPELINE_VERSION)
        self.assertTrue(run["preprocessing"]["normalise"])
        self.assertEqual(run["preprocessing"]["censored_policy"], "skip")
        self.assertEqual(run["preprocessing"]["weighting_mode"], "equal")
        self.assertEqual(run["preprocessing"]["elements_used"], ["Au", "Cu", "Zn"])
        self.assertEqual(run["preprocessing"]["n_shared_elements"], 3)
        self.assertEqual(run["reference_library_version"], "osnaca-2026-05")
        self.assertEqual(run["dataset_id"], 12)
        self.assertIsInstance(run["runtime_ms"], float)
        self.assertGreaterEqual(run["runtime_ms"], 0.0)

    def test_top_n_limits_the_ranked_matches(self):
        envelope = get_algorithm("log_difference_similarity").compare(
            SAMPLES,
            REFERENCES,
            {"top_n": 2, "preprocessing": {}},
        ).to_envelope()

        self.assertEqual(len(envelope["matches"]), 2)

    def test_an_empty_reference_library_warns_rather_than_failing(self):
        envelope = get_algorithm("log_difference_similarity").compare(
            SAMPLES,
            [],
            {"top_n": 10},
        ).to_envelope()

        self.assertEqual(envelope["matches"], [])
        self.assertEqual(validate_envelope(envelope), [])
        self.assertTrue(envelope["warnings"])

    def test_references_sharing_no_elements_are_skipped(self):
        envelope = get_algorithm("log_difference_similarity").compare(
            SAMPLES,
            [{"id": 1, "values": {"Pb": 5.0, "Ni": 3.0}}],
            {"top_n": 10},
        ).to_envelope()

        self.assertEqual(envelope["matches"], [])

    def test_optional_blocks_are_omitted_rather_than_sent_as_null(self):
        """Contract rule 2 — consumers test for presence, not for emptiness."""
        envelope = self.envelope_for("log_difference_similarity")

        self.assertNotIn("projection", envelope)
        self.assertNotIn("evidence", envelope["matches"][0])


class EnvelopeValidatorTests(SimpleTestCase):
    """
    The validator has to actually reject things.

    A contract checker that passes everything is worse than none, because it
    creates confidence that nothing has been verified.
    """

    def valid_result(self, **overrides):
        defaults = {
            "algorithm_id": "log_difference_similarity",
            "algorithm_version": "1.0.0",
            "matches": [Match(rank=1, similarity=0.87, reference_sample_id=1)],
            "runtime_ms": 12.5,
        }
        return RunResult(**{**defaults, **overrides})

    def test_a_well_formed_envelope_passes(self):
        self.assertEqual(validate_envelope(self.valid_result().to_envelope()), [])

    def test_a_similarity_above_one_is_rejected(self):
        envelope = self.valid_result(
            matches=[Match(rank=1, similarity=1.4)],
        ).to_envelope()

        problems = validate_envelope(envelope)

        self.assertEqual(len(problems), 1)
        self.assertIn("within [0, 1]", problems[0])

    def test_a_negative_similarity_is_rejected(self):
        envelope = self.valid_result(
            matches=[Match(rank=1, similarity=-0.1)],
        ).to_envelope()

        self.assertIn("within [0, 1]", validate_envelope(envelope)[0])

    def test_a_missing_similarity_is_rejected(self):
        envelope = self.valid_result().to_envelope()
        del envelope["matches"][0]["scores"]["similarity"]

        self.assertIn("similarity is missing", validate_envelope(envelope)[0])

    def test_a_wrong_schema_version_is_rejected(self):
        envelope = self.valid_result().to_envelope()
        envelope["schema_version"] = "0.9"

        self.assertIn("schema_version", validate_envelope(envelope)[0])

    def test_a_missing_run_block_is_rejected(self):
        envelope = self.valid_result().to_envelope()
        del envelope["run"]

        self.assertIn("run block is missing", validate_envelope(envelope))

    def test_an_incomplete_evidence_entry_is_rejected(self):
        envelope = self.valid_result().to_envelope()
        envelope["matches"][0]["evidence"] = {
            "supporting": [{"element": "Zn", "contribution": 0.21}],
            "conflicting": [],
        }

        self.assertIn("imputed", validate_envelope(envelope)[0])

    def test_raw_metrics_travel_beside_the_normalised_score(self):
        """Contract rule 1 — algorithm-native metrics keep their own names."""
        envelope = self.valid_result(
            matches=[Match(
                rank=1,
                similarity=0.87,
                scores={"aitchison_distance": 2.31, "spearman_rho": 0.79},
            )],
        ).to_envelope()

        self.assertEqual(validate_envelope(envelope), [])
        self.assertEqual(
            envelope["matches"][0]["scores"],
            {"similarity": 0.87, "aitchison_distance": 2.31, "spearman_rho": 0.79},
        )

    def test_signed_evidence_serialises_in_the_shared_shape(self):
        """Contract rule 3 — one shape whether it came from a distance or SHAP."""
        envelope = self.valid_result(
            matches=[Match(
                rank=1,
                similarity=0.87,
                supporting=[Evidence("Zn", 0.21)],
                conflicting=[Evidence("Cr", -0.12, imputed=True)],
            )],
        ).to_envelope()

        self.assertEqual(validate_envelope(envelope), [])
        self.assertEqual(
            envelope["matches"][0]["evidence"],
            {
                "supporting": [
                    {"element": "Zn", "contribution": 0.21, "imputed": False},
                ],
                "conflicting": [
                    {"element": "Cr", "contribution": -0.12, "imputed": True},
                ],
            },
        )
