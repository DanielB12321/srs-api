"""
Characterisation tests: the regression net for the pluggable-algorithm work.

These pin the behaviour of the similarity scoring and ranking as it exists
before any algorithm is extracted out of the view. Every later refactor must
keep them passing unchanged; a failure means observable behaviour moved, not
that the test needs updating.

Scores are compared to twelve decimal places. That is tight enough to catch any
change to the arithmetic while tolerating the last-bit reassociation that comes
from summing the same terms in a different order.

The scoring functions are exercised directly rather than through the HTTP
endpoint, because a POST returns 202 and finishes the work on a background
thread. Calling through the endpoint would race the assertions.
"""

from django.test import SimpleTestCase, TestCase

from ..api_views import FullAnalysisListCreateView
from ..models import (
    Element,
    FullAnalysis,
    FullAnalysisMatch,
    ReferenceDeposit,
    ReferenceSample,
    ReferenceSampleMeasurement,
)

# One tested sample and three references chosen so the expected scores can be
# derived by hand: an exact match, a single element moved one decade, and two
# elements swapped across a decade in opposite directions.
INPUT_VALUES = {"Cu": 100.0, "Zn": 10.0, "Au": 1.0}
REFERENCE_VALUES = {
    "identical": {"Cu": 100.0, "Zn": 10.0, "Au": 1.0},
    "cu_x10": {"Cu": 1000.0, "Zn": 10.0, "Au": 1.0},
    "swapped": {"Cu": 10.0, "Zn": 100.0, "Au": 1.0},
}
COMMON_ELEMENTS = {"Cu", "Zn", "Au"}

NO_PREPROCESSING = {}
LOG_TRANSFORM = {"log_transform": True}
NORMALISE = {"normalise": True}

# method -> preprocessing -> reference -> score.
#
# log_difference_similarity scores 1 for an identical element and 0.5 for a
# tenfold difference, so cu_x10 is (0.5 + 1 + 1) / 3 and swapped is
# (0.5 + 0.5 + 1) / 3. Under CLR the vectors are centred on their geometric
# mean first, which is why normalise gives a different answer.
#
# distance applies the same 1 / (1 + |difference|) curve to whatever the
# preprocessing stage produced, so without preprocessing it runs on raw ppm and
# a 900 ppm gap collapses almost to zero. With log_transform it coincides
# exactly with log_difference_similarity, which is the relationship the
# Aitchison work in a later phase builds on.
EXPECTED_SCORES = {
    "log_difference_similarity": {
        "no_preprocessing": {
            "identical": 1.0,
            "cu_x10": 0.8333333333333334,
            "swapped": 0.6666666666666666,
        },
        "log_transform": {
            "identical": 1.0,
            "cu_x10": 0.8333333333333334,
            "swapped": 0.6666666666666666,
        },
        "normalise": {
            "identical": 1.0,
            "cu_x10": 0.7000000000000001,
            "swapped": 0.6666666666666666,
        },
    },
    "correlation": {
        "no_preprocessing": {
            "identical": 1.0,
            "cu_x10": 0.9986147470400143,
            "swapped": 0.32432432432432434,
        },
        "log_transform": {
            "identical": 1.0,
            "cu_x10": 0.9909902530309829,
            "swapped": 0.75,
        },
        "normalise": {
            "identical": 1.0,
            "cu_x10": 0.9909902530309829,
            "swapped": 0.75,
        },
    },
    "association": {
        "no_preprocessing": {
            "identical": 1,
            "cu_x10": 0.9979712892921635,
            "swapped": 0.5990495990495991,
        },
        "log_transform": {
            "identical": 1,
            "cu_x10": 0.9949747468305832,
            "swapped": 0.9,
        },
        "normalise": {
            "identical": 1,
            "cu_x10": 0.9909902530309829,
            "swapped": 0.75,
        },
    },
    "distance": {
        "no_preprocessing": {
            "identical": 1.0,
            "cu_x10": 0.6670366259711432,
            "swapped": 0.3406593406593406,
        },
        "log_transform": {
            "identical": 1.0,
            "cu_x10": 0.8333333333333334,
            "swapped": 0.6666666666666666,
        },
        "normalise": {
            "identical": 1.0,
            "cu_x10": 0.7000000000000001,
            "swapped": 0.6666666666666666,
        },
    },
}

PREPROCESSING_STATES = {
    "no_preprocessing": NO_PREPROCESSING,
    "log_transform": LOG_TRANSFORM,
    "normalise": NORMALISE,
}


class SimilarityScoreCharacterisationTests(SimpleTestCase):
    """Pin the scoring arithmetic for every method and preprocessing state."""

    def setUp(self):
        # Instantiating the view without a request is deliberate. The scoring
        # code must stay callable from a management command or worker, so it
        # may never reach for self.request.
        self.view = FullAnalysisListCreateView()

    def score(self, reference_key, method, preprocessing):
        return self.view.calculate_similarity_score(
            INPUT_VALUES,
            REFERENCE_VALUES[reference_key],
            COMMON_ELEMENTS,
            method,
            preprocessing,
        )

    def test_every_method_and_preprocessing_combination(self):
        for method, by_preprocessing in EXPECTED_SCORES.items():
            for state_name, expected_by_reference in by_preprocessing.items():
                for reference_key, expected in expected_by_reference.items():
                    with self.subTest(
                        method=method,
                        preprocessing=state_name,
                        reference=reference_key,
                    ):
                        self.assertAlmostEqual(
                            self.score(
                                reference_key,
                                method,
                                PREPROCESSING_STATES[state_name],
                            ),
                            expected,
                            places=12,
                        )

    def test_unknown_method_falls_back_to_log_difference(self):
        """An unrecognised method must not error or score zero."""
        for state_name, preprocessing in PREPROCESSING_STATES.items():
            for reference_key in REFERENCE_VALUES:
                with self.subTest(preprocessing=state_name, reference=reference_key):
                    self.assertAlmostEqual(
                        self.score(reference_key, "not_a_real_method", preprocessing),
                        EXPECTED_SCORES["log_difference_similarity"][state_name][
                            reference_key
                        ],
                        places=12,
                    )

    def test_missing_method_and_preprocessing_arguments_are_optional(self):
        self.assertAlmostEqual(
            self.view.calculate_similarity_score(
                INPUT_VALUES,
                REFERENCE_VALUES["cu_x10"],
                COMMON_ELEMENTS,
            ),
            0.8333333333333334,
            places=12,
        )

    def test_no_shared_elements_scores_zero(self):
        self.assertEqual(
            self.view.calculate_similarity_score(INPUT_VALUES, {}, set()),
            0,
        )

    def test_every_score_is_within_the_zero_to_one_range(self):
        """Envelope rule 1 already holds for the existing methods."""
        for method, by_preprocessing in EXPECTED_SCORES.items():
            for state_name, preprocessing in PREPROCESSING_STATES.items():
                for reference_key in REFERENCE_VALUES:
                    with self.subTest(
                        method=method,
                        preprocessing=state_name,
                        reference=reference_key,
                    ):
                        score = self.score(reference_key, method, preprocessing)
                        self.assertGreaterEqual(score, 0.0)
                        self.assertLessEqual(score, 1.0)

    def test_association_returns_an_integer_for_an_exact_match(self):
        """
        Known quirk. min/max over integer bounds hands back int 1 rather than
        1.0 for a perfect match. The envelope requires a float, so whichever
        adapter wraps this method has to coerce the result.
        """
        score = self.score("identical", "association", NO_PREPROCESSING)
        self.assertEqual(score, 1)
        self.assertIsInstance(score, int)


class RankedMatchCharacterisationTests(TestCase):
    """Pin ranking, tie-breaking, truncation, and the measurement filters."""

    def setUp(self):
        self.view = FullAnalysisListCreateView()
        self.elements = {
            symbol: Element.objects.create(symbol=symbol)
            for symbol in ("Cu", "Zn", "Au")
        }
        self.deposit = ReferenceDeposit.objects.create(
            name="Alpha",
            three_char_code="ALP",
            deposit_type="VHMS",
        )
        self.full_analysis = FullAnalysis.objects.create(
            name="Characterisation analysis",
            uploaded_sample_code="S001",
            method="log_difference_similarity",
        )

    def add_reference_sample(self, sample_code, values, unit="ppm", below_dl=False):
        """Create one reference sample from a symbol-to-value mapping."""
        reference_sample = ReferenceSample.objects.create(
            reference_deposit=self.deposit,
            sample_code=sample_code,
        )
        for symbol, value in values.items():
            ReferenceSampleMeasurement.objects.create(
                reference_sample=reference_sample,
                element=self.elements[symbol],
                value=value,
                unit=unit,
                below_detection_limit=below_dl,
            )
        return reference_sample

    def input_measurements(self, values=None, unit="ppm", below_dl=False):
        return [
            {
                "element_symbol": symbol,
                "value": value,
                "unit": unit,
                "below_detection_limit": below_dl,
                "detection_limit": None,
            }
            for symbol, value in (values or INPUT_VALUES).items()
        ]

    def rank(self, measurements=None, top_n=10, batch_size=250):
        self.view.create_ranked_matches(
            self.full_analysis,
            measurements if measurements is not None else self.input_measurements(),
            0,
            top_n,
            batch_size,
            "log_difference_similarity",
            {},
        )
        return list(
            FullAnalysisMatch.objects
            .filter(full_analysis=self.full_analysis, analysed_sample_index=0)
            .order_by("rank")
        )

    def test_ranking_order_scores_and_tie_break(self):
        first_exact = self.add_reference_sample("R1", REFERENCE_VALUES["identical"])
        one_decade = self.add_reference_sample("R2", REFERENCE_VALUES["cu_x10"])
        two_decades = self.add_reference_sample("R3", REFERENCE_VALUES["swapped"])
        second_exact = self.add_reference_sample("R4", REFERENCE_VALUES["identical"])

        matches = self.rank()

        # Ties are broken by reference sample ID descending, because candidates
        # are pushed onto the heap as (score, reference_sample_id) tuples and
        # then sorted in reverse. R4 outranks R1 on an identical score.
        self.assertEqual(
            [match.reference_sample_id for match in matches],
            [
                second_exact.id,
                first_exact.id,
                one_decade.id,
                two_decades.id,
            ],
        )
        self.assertEqual([match.rank for match in matches], [1, 2, 3, 4])
        for match, expected in zip(
            matches,
            [1.0, 1.0, 0.8333333333333334, 0.6666666666666666],
        ):
            self.assertAlmostEqual(match.similarity_score, expected, places=12)

    def test_top_n_truncates_to_the_highest_scoring_references(self):
        self.add_reference_sample("R1", REFERENCE_VALUES["identical"])
        self.add_reference_sample("R2", REFERENCE_VALUES["cu_x10"])
        self.add_reference_sample("R3", REFERENCE_VALUES["swapped"])

        matches = self.rank(top_n=2)

        self.assertEqual(len(matches), 2)
        for match, expected in zip(matches, [1.0, 0.8333333333333334]):
            self.assertAlmostEqual(match.similarity_score, expected, places=12)

    def test_batch_size_does_not_change_the_final_ranking(self):
        """
        The heap is kept across batches, so a small batch size must produce the
        same global ranking rather than a per-batch one.
        """
        self.add_reference_sample("R1", REFERENCE_VALUES["identical"])
        self.add_reference_sample("R2", REFERENCE_VALUES["cu_x10"])
        self.add_reference_sample("R3", REFERENCE_VALUES["swapped"])

        single_batch = [
            (match.reference_sample_id, match.similarity_score)
            for match in self.rank(batch_size=250)
        ]
        many_batches = [
            (match.reference_sample_id, match.similarity_score)
            for match in self.rank(batch_size=1)
        ]

        self.assertEqual(single_batch, many_batches)

    def test_units_are_converted_before_comparison(self):
        """100000 ppb is 100 ppm, so this reference must score an exact match."""
        self.add_reference_sample(
            "R_ppb",
            {"Cu": 100000.0, "Zn": 10000.0, "Au": 1000.0},
            unit="ppb",
        )

        matches = self.rank()

        self.assertEqual(len(matches), 1)
        self.assertAlmostEqual(matches[0].similarity_score, 1.0, places=12)

    def test_unsupported_units_are_excluded_rather_than_compared(self):
        self.add_reference_sample(
            "R_bad_unit",
            REFERENCE_VALUES["identical"],
            unit="mol/L",
        )

        self.assertEqual(self.rank(), [])

    def test_below_detection_limit_measurements_are_skipped(self):
        self.add_reference_sample(
            "R_bdl",
            REFERENCE_VALUES["identical"],
            below_dl=True,
        )
        self.add_reference_sample("R_ok", REFERENCE_VALUES["cu_x10"])

        matches = self.rank()

        # The below-detection reference shares no usable element, so it is not
        # ranked at all rather than being ranked with a low score.
        self.assertEqual(len(matches), 1)
        self.assertAlmostEqual(
            matches[0].similarity_score,
            0.8333333333333334,
            places=12,
        )

    def test_non_positive_and_null_input_values_are_skipped(self):
        self.add_reference_sample("R1", {"Zn": 10.0, "Au": 1.0})

        matches = self.rank(
            measurements=self.input_measurements(
                {"Cu": 0.0, "Zn": 10.0, "Au": 1.0},
            ),
        )

        # Cu drops out of the comparison entirely, leaving two exact elements.
        self.assertEqual(len(matches), 1)
        self.assertAlmostEqual(matches[0].similarity_score, 1.0, places=12)

    def test_rerunning_replaces_rather_than_appends_matches(self):
        self.add_reference_sample("R1", REFERENCE_VALUES["identical"])

        self.rank()
        matches = self.rank()

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].rank, 1)

    def test_progress_counters_are_written_to_parameters(self):
        self.add_reference_sample("R1", REFERENCE_VALUES["identical"])
        self.add_reference_sample("R2", REFERENCE_VALUES["cu_x10"])

        self.rank(batch_size=1)
        self.full_analysis.refresh_from_db()

        self.assertEqual(self.full_analysis.parameters["reference_count"], 2)
        self.assertEqual(self.full_analysis.parameters["references_processed"], 2)
        self.assertEqual(self.full_analysis.parameters["current_sample_index"], 0)
