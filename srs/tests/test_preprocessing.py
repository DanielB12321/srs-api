"""Tests for measurement preprocessing and its analysis options."""

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
from ..preprocessing import (
    DETECTION_LIMIT,
    EQUAL,
    HALF_DETECTION_LIMIT,
    PIPELINE_VERSION,
    SELECTED_BOOST,
    SKIP,
    describe,
    extract_values,
    normalise_symbol,
    prepare_vectors,
    resolve_options,
)
from .test_characterisation import INPUT_VALUES, REFERENCE_VALUES


def measurement(symbol, value, unit="ppm", below_dl=False, detection_limit=None):
    return {
        "element_symbol": symbol,
        "value": value,
        "unit": unit,
        "below_detection_limit": below_dl,
        "detection_limit": detection_limit,
    }


class OptionResolutionTests(SimpleTestCase):
    """Check defaults and request option validation."""

    def test_defaults_reproduce_the_behaviour_that_existed_before(self):
        options = resolve_options()

        self.assertFalse(options["normalise"])
        self.assertFalse(options["log_transform"])
        self.assertEqual(options["handle_missing"], SKIP)
        self.assertEqual(options["weighting_mode"], EQUAL)
        self.assertEqual(options["selected_elements"], [])

    def test_unknown_policy_names_fall_back_instead_of_raising(self):
        options = resolve_options({
            "handle_missing": "something_invented_later",
            "weighting_mode": "also_unknown",
        })

        self.assertEqual(options["handle_missing"], SKIP)
        self.assertEqual(options["weighting_mode"], EQUAL)

    def test_recognised_policies_are_kept(self):
        options = resolve_options({
            "handle_missing": HALF_DETECTION_LIMIT,
            "weighting_mode": SELECTED_BOOST,
            "normalise": True,
        })

        self.assertEqual(options["handle_missing"], HALF_DETECTION_LIMIT)
        self.assertEqual(options["weighting_mode"], SELECTED_BOOST)
        self.assertTrue(options["normalise"])

    def test_clr_disables_the_separate_log_transform(self):
        options = resolve_options({
            "normalise": True,
            "log_transform": True,
        })

        self.assertTrue(options["normalise"])
        self.assertFalse(options["log_transform"])
        self.assertFalse(describe(options)["log_transform"])

    def test_selected_element_symbols_are_normalised(self):
        options = resolve_options({}, ["cu", "ZN", " au ", "", None])

        self.assertEqual(options["selected_elements"], ["Cu", "Zn", "Au"])

    def test_symbol_normalisation_handles_the_awkward_cases(self):
        self.assertEqual(normalise_symbol("CU"), "Cu")
        self.assertEqual(normalise_symbol(" au "), "Au")
        self.assertEqual(normalise_symbol("c"), "C")
        self.assertEqual(normalise_symbol(""), "")
        self.assertEqual(normalise_symbol(None), "")


class CensoredValueTests(SimpleTestCase):
    """Check the supported below-detection-limit policies."""

    def censored_sample(self, unit="ppm"):
        return [
            measurement("Cu", 100.0, unit=unit),
            measurement("Zn", None, unit=unit, below_dl=True, detection_limit=4.0),
        ]

    def test_skip_discards_censored_measurements(self):
        values, imputed = extract_values(
            self.censored_sample(),
            resolve_options({"handle_missing": SKIP}),
        )

        self.assertEqual(values, {"Cu": 100.0})
        self.assertEqual(imputed, set())

    def test_half_detection_limit_substitutes_and_marks_the_value(self):
        values, imputed = extract_values(
            self.censored_sample(),
            resolve_options({"handle_missing": HALF_DETECTION_LIMIT}),
        )

        self.assertEqual(values, {"Cu": 100.0, "Zn": 2.0})
        self.assertEqual(imputed, {"Zn"})

    def test_detection_limit_substitutes_the_limit_itself(self):
        values, imputed = extract_values(
            self.censored_sample(),
            resolve_options({"handle_missing": DETECTION_LIMIT}),
        )

        self.assertEqual(values, {"Cu": 100.0, "Zn": 4.0})
        self.assertEqual(imputed, {"Zn"})

    def test_the_detection_limit_is_converted_to_ppm_like_any_other_value(self):
        values, _ = extract_values(
            [measurement("Au", None, unit="ppb", below_dl=True, detection_limit=5.0)],
            resolve_options({"handle_missing": HALF_DETECTION_LIMIT}),
        )

        self.assertAlmostEqual(values["Au"], 0.0025, places=12)

    def test_a_censored_value_without_a_limit_is_still_skipped(self):
        values, imputed = extract_values(
            [measurement("Zn", None, below_dl=True, detection_limit=None)],
            resolve_options({"handle_missing": HALF_DETECTION_LIMIT}),
        )

        self.assertEqual(values, {})
        self.assertEqual(imputed, set())


class ExtractionTests(SimpleTestCase):
    """Check unit conversion and invalid-value filtering."""

    def test_units_are_converted_to_ppm(self):
        values, _ = extract_values([
            measurement("Au", 1000.0, unit="ppb"),
            measurement("Fe", 3.2, unit="pct"),
            measurement("Cu", 55.0, unit="ppm"),
        ])

        self.assertAlmostEqual(values["Au"], 1.0, places=12)
        self.assertAlmostEqual(values["Fe"], 32000.0, places=12)
        self.assertAlmostEqual(values["Cu"], 55.0, places=12)

    def test_unsupported_units_and_non_positive_values_are_left_out(self):
        values, _ = extract_values([
            measurement("Cu", 10.0, unit="mol/L"),
            measurement("Zn", 0.0),
            measurement("Pb", -5.0),
            measurement("Au", None),
            measurement("Ag", 1.0),
        ])

        self.assertEqual(set(values), {"Ag"})

    def test_symbols_are_normalised_during_extraction(self):
        values, _ = extract_values([measurement("cu", 10.0)])

        self.assertEqual(set(values), {"Cu"})


class SelectedElementTests(SimpleTestCase):
    """Check filtering and boost behaviour for selected elements."""

    def sample(self):
        return [
            measurement("Cu", 100.0),
            measurement("Zn", 10.0),
            measurement("Au", 1.0),
        ]

    def test_an_empty_selection_keeps_every_element(self):
        values, _ = extract_values(self.sample(), resolve_options({}, []))

        self.assertEqual(set(values), {"Cu", "Zn", "Au"})

    def test_a_selection_restricts_the_comparison(self):
        values, _ = extract_values(self.sample(), resolve_options({}, ["Cu", "Au"]))

        self.assertEqual(set(values), {"Cu", "Au"})

    def test_selection_matching_is_case_insensitive(self):
        values, _ = extract_values(self.sample(), resolve_options({}, ["cu", "AU"]))

        self.assertEqual(set(values), {"Cu", "Au"})

    def test_selected_boost_keeps_every_element_instead_of_filtering(self):
        options = resolve_options({"weighting_mode": SELECTED_BOOST}, ["Cu"])
        values, _ = extract_values(self.sample(), options)

        self.assertEqual(set(values), {"Cu", "Zn", "Au"})


class WeightingTests(SimpleTestCase):
    """Check selected-element weights and the unweighted default."""

    def prepared(self, preprocessing=None, selected=None):
        return prepare_vectors(
            INPUT_VALUES,
            REFERENCE_VALUES["identical"],
            {"Cu", "Zn", "Au"},
            resolve_options(preprocessing, selected),
        )

    def test_equal_weighting_produces_no_weights_at_all(self):
        self.assertIsNone(self.prepared().weights)

    def test_selected_boost_weights_the_chosen_elements_higher(self):
        prepared = self.prepared({"weighting_mode": SELECTED_BOOST}, ["Cu"])

        self.assertEqual(prepared.symbols, ["Au", "Cu", "Zn"])
        self.assertEqual(prepared.weights, [1.0, 2.0, 1.0])

    def test_boosting_every_element_is_the_same_as_boosting_none(self):
        prepared = self.prepared(
            {"weighting_mode": SELECTED_BOOST},
            ["Cu", "Zn", "Au"],
        )

        self.assertIsNone(prepared.weights)

    def test_boosting_without_a_selection_does_nothing(self):
        prepared = self.prepared({"weighting_mode": SELECTED_BOOST}, [])

        self.assertIsNone(prepared.weights)


class PreparedVectorTests(SimpleTestCase):
    """Check vector alignment, transforms and imputed flags."""

    def test_vectors_are_aligned_by_sorted_symbol(self):
        prepared = prepare_vectors(
            INPUT_VALUES,
            REFERENCE_VALUES["cu_x10"],
            {"Cu", "Zn", "Au"},
        )

        self.assertEqual(prepared.symbols, ["Au", "Cu", "Zn"])
        self.assertEqual(prepared.input_vector, [1.0, 100.0, 10.0])
        self.assertEqual(prepared.reference_vector, [1.0, 1000.0, 10.0])

    def test_the_imputed_mask_lines_up_with_the_symbols(self):
        prepared = prepare_vectors(
            INPUT_VALUES,
            REFERENCE_VALUES["identical"],
            {"Cu", "Zn", "Au"},
            resolve_options(),
            imputed_elements={"Cu"},
        )

        self.assertEqual(prepared.symbols, ["Au", "Cu", "Zn"])
        self.assertEqual(prepared.imputed, [False, True, False])

    def test_nothing_is_imputed_by_default(self):
        prepared = prepare_vectors(
            INPUT_VALUES,
            REFERENCE_VALUES["identical"],
            {"Cu", "Zn", "Au"},
        )

        self.assertEqual(prepared.imputed, [False, False, False])

    def test_log_transform_puts_the_vectors_into_log_space(self):
        prepared = prepare_vectors(
            INPUT_VALUES,
            REFERENCE_VALUES["identical"],
            {"Cu", "Zn", "Au"},
            resolve_options({"log_transform": True}),
        )

        self.assertTrue(prepared.in_log_space)
        self.assertEqual(prepared.input_vector, [0.0, 2.0, 1.0])

    def test_clr_centres_each_sample_on_its_own_geometric_mean(self):
        prepared = prepare_vectors(
            INPUT_VALUES,
            REFERENCE_VALUES["identical"],
            {"Cu", "Zn", "Au"},
            resolve_options({"normalise": True}),
        )

        self.assertTrue(prepared.in_log_space)
        # log10 of 1, 100, 10 is 0, 2, 1; the mean is 1, so centring gives -1, 1, 0.
        self.assertEqual(prepared.input_vector, [-1.0, 1.0, 0.0])
        self.assertAlmostEqual(sum(prepared.input_vector), 0.0, places=12)

    def test_untransformed_vectors_are_not_in_log_space(self):
        self.assertFalse(prepare_vectors(
            INPUT_VALUES,
            REFERENCE_VALUES["identical"],
            {"Cu", "Zn", "Au"},
        ).in_log_space)


class DescribeTests(SimpleTestCase):
    """Check the preprocessing summary stored with an analysis."""

    def test_the_description_carries_the_pipeline_version_and_policies(self):
        described = describe(
            resolve_options(
                {"handle_missing": HALF_DETECTION_LIMIT, "normalise": True},
                ["Cu", "Zn"],
            ),
            ["Au", "Cu", "Zn"],
        )

        self.assertEqual(described["pipeline_version"], PIPELINE_VERSION)
        self.assertEqual(described["censored_policy"], HALF_DETECTION_LIMIT)
        self.assertTrue(described["normalise"])
        self.assertFalse(described["log_transform"])
        self.assertEqual(described["selected_elements"], ["Cu", "Zn"])
        self.assertEqual(described["elements_used"], ["Au", "Cu", "Zn"])
        self.assertEqual(described["n_shared_elements"], 3)


class TogglesReachTheRankingPathTests(TestCase):
    """Check preprocessing options through the stored ranking path."""

    def setUp(self):
        self.view = FullAnalysisListCreateView()
        self.elements = {
            symbol: Element.objects.create(symbol=symbol)
            for symbol in ("Cu", "Zn", "Au")
        }
        self.deposit = ReferenceDeposit.objects.create(
            name="Alpha",
            three_char_code="ALP",
        )

    def add_reference(self, sample_code, values, censored=()):
        reference_sample = ReferenceSample.objects.create(
            reference_deposit=self.deposit,
            sample_code=sample_code,
        )
        for symbol, value in values.items():
            is_censored = symbol in censored
            ReferenceSampleMeasurement.objects.create(
                reference_sample=reference_sample,
                element=self.elements[symbol],
                value=None if is_censored else value,
                unit="ppm",
                below_detection_limit=is_censored,
                detection_limit=value if is_censored else None,
            )
        return reference_sample

    def analysis(self, selected_elements=None):
        return FullAnalysis.objects.create(
            name="Toggle analysis",
            uploaded_sample_code="S001",
            parameters={"selected_elements": selected_elements or []},
        )

    def rank(self, full_analysis, measurements, preprocessing):
        self.view.create_ranked_matches(
            full_analysis,
            measurements,
            0,
            10,
            250,
            "log_difference_similarity",
            preprocessing,
        )
        return list(
            FullAnalysisMatch.objects
            .filter(full_analysis=full_analysis, analysed_sample_index=0)
            .order_by("rank")
        )

    def input_sample(self):
        return [measurement(symbol, value) for symbol, value in INPUT_VALUES.items()]

    def test_skip_ignores_a_censored_reference_element(self):
        self.add_reference("R1", REFERENCE_VALUES["cu_x10"], censored={"Cu"})

        matches = self.rank(self.analysis(), self.input_sample(), {})

        self.assertAlmostEqual(matches[0].similarity_score, 1.0, places=12)

    def test_half_detection_limit_brings_the_censored_element_back(self):
        self.add_reference("R1", REFERENCE_VALUES["cu_x10"], censored={"Cu"})

        matches = self.rank(
            self.analysis(),
            self.input_sample(),
            {"handle_missing": HALF_DETECTION_LIMIT},
        )

        # Cu becomes 500 against an input of 100, which is 0.69897 decades
        # apart, so that element scores 1 / 1.69897.
        expected = (1 / (1 + abs(2.69897000433602 - 2.0)) + 1 + 1) / 3
        self.assertAlmostEqual(matches[0].similarity_score, expected, places=10)

    def test_the_element_selection_restricts_the_comparison(self):
        self.add_reference("R1", REFERENCE_VALUES["cu_x10"])

        without_selection = self.rank(self.analysis(), self.input_sample(), {})
        with_selection = self.rank(
            self.analysis(selected_elements=["Zn", "Au"]),
            self.input_sample(),
            {},
        )

        self.assertAlmostEqual(
            without_selection[0].similarity_score,
            0.8333333333333334,
            places=12,
        )
        self.assertAlmostEqual(with_selection[0].similarity_score, 1.0, places=12)

    def test_selected_boost_reweights_without_excluding_anything(self):
        self.add_reference("R1", REFERENCE_VALUES["cu_x10"])

        matches = self.rank(
            self.analysis(selected_elements=["Cu"]),
            self.input_sample(),
            {"weighting_mode": SELECTED_BOOST},
        )

        # Cu scores 0.5 at double weight, Zn and Au score 1 at single weight.
        self.assertAlmostEqual(
            matches[0].similarity_score,
            (2 * 0.5 + 1 + 1) / 4,
            places=12,
        )

    def test_default_options_leave_the_score_exactly_where_it_was(self):
        self.add_reference("R1", REFERENCE_VALUES["swapped"])

        matches = self.rank(self.analysis(), self.input_sample(), {})

        self.assertAlmostEqual(
            matches[0].similarity_score,
            0.6666666666666666,
            places=12,
        )
