"""Tests for analysis-wide element relationships."""

from django.test import SimpleTestCase, TestCase

from ..api_views import FullAnalysisListCreateView
from ..associations import (
    _average_ranks,
    _two_sided_p_value,
    calculate_element_associations,
    resolve_association_options,
)
from ..models import FullAnalysis


def sample(code, au, cu, pb=None):
    measurements = [
        {"element_symbol": "Au", "value": au, "unit": "ppm"},
        {"element_symbol": "Cu", "value": cu, "unit": "ppm"},
    ]
    if pb is not None:
        measurements.append(
            {"element_symbol": "Pb", "value": pb, "unit": "ppm"}
        )
    return {"sample_code": code, "measurements": measurements}


class ElementAssociationTests(SimpleTestCase):
    def setUp(self):
        self.samples = [
            sample("S1", 1, 2, 8),
            sample("S2", 2, 4, 6),
            sample("S3", 3, 6, 4),
            sample("S4", 4, 8, 2),
        ]

    def test_positive_and_negative_relationships_are_found(self):
        result = calculate_element_associations(self.samples)
        pairs = {
            (row["element_a"], row["element_b"]): row
            for row in result["associations"]
        }

        self.assertEqual(pairs[("Au", "Cu")]["correlation"], 1.0)
        self.assertEqual(pairs[("Au", "Pb")]["correlation"], -1.0)
        self.assertEqual(pairs[("Au", "Pb")]["direction"], "negative")
        self.assertEqual(pairs[("Au", "Pb")]["strength"], "strong")
        self.assertEqual(pairs[("Au", "Pb")]["shared_sample_count"], 4)

    def test_selected_elements_limit_the_association_result(self):
        result = calculate_element_associations(
            self.samples,
            selected_elements=["Au", "Cu"],
        )

        self.assertEqual(result["eligible_elements"], ["Au", "Cu"])
        self.assertEqual(len(result["associations"]), 1)

    def test_each_pair_records_its_actual_shared_sample_count(self):
        samples = self.samples[:3] + [sample("S4", 4, 8)]

        result = calculate_element_associations(samples)
        au_pb = next(
            row for row in result["associations"]
            if {row["element_a"], row["element_b"]} == {"Au", "Pb"}
        )

        self.assertEqual(au_pb["shared_sample_count"], 3)

    def test_fewer_than_three_samples_returns_a_clear_message(self):
        result = calculate_element_associations(self.samples[:2])

        self.assertFalse(result["available"])
        self.assertEqual(result["associations"], [])
        self.assertIn("At least three", result["message"])

    def test_clr_setting_is_used_and_recorded(self):
        result = calculate_element_associations(
            self.samples,
            preprocessing={"normalise": True},
        )

        self.assertEqual(result["value_space"], "clr")
        self.assertTrue(result["available"])

    def test_spearman_handles_monotonic_values_and_tied_ranks(self):
        samples = [
            sample(f"S{index}", index, 2 ** index)
            for index in range(1, 11)
        ]

        spearman = calculate_element_associations(
            samples,
            association_options={
                "method": "spearman",
                "minimum_shared_samples": 5,
                "minimum_absolute_correlation": 0.7,
                "maximum_adjusted_p_value": 0.05,
            },
        )
        pearson = calculate_element_associations(
            samples,
            association_options={"method": "pearson"},
        )

        self.assertEqual(_average_ranks([10, 10, 20]), [1.5, 1.5, 3.0])
        self.assertEqual(spearman["associations"][0]["correlation"], 1.0)
        self.assertGreater(
            spearman["associations"][0]["correlation"],
            pearson["associations"][0]["correlation"],
        )

    def test_p_values_match_known_correlation_examples(self):
        self.assertAlmostEqual(_two_sided_p_value(0.5, 10), 0.141113, places=5)
        self.assertAlmostEqual(_two_sided_p_value(0.9, 10), 0.000387, places=5)
        self.assertEqual(_two_sided_p_value(1.0, 10), 0.0)

    def test_filters_use_adjusted_p_value_strength_and_sample_count(self):
        result = calculate_element_associations(
            self.samples,
            association_options={
                "method": "pearson",
                "minimum_shared_samples": 5,
                "minimum_absolute_correlation": 0.7,
                "maximum_adjusted_p_value": 0.05,
            },
        )

        self.assertTrue(result["available"])
        self.assertEqual(result["filtered_association_count"], 0)
        self.assertTrue(all(
            row["adjusted_p_value"] >= row["p_value"]
            for row in result["associations"]
        ))
        self.assertTrue(all(
            row["reliability"] == "limited"
            for row in result["associations"]
        ))

    def test_geochemical_style_missing_and_extreme_values_remain_usable(self):
        gold = [0.01, 0.03, 0.08, 0.2, 0.7, 1.8, 5.0, 20.0, 120.0, 900.0]
        arsenic = [2, 3, 5, 9, 16, 30, 55, 110, 250, 600]
        copper = [35, 12, 70, 20, 55, 18, 90, 25, 45, 30]
        samples = []
        for index, (au, arsenic_value, copper_value) in enumerate(
            zip(gold, arsenic, copper),
            start=1,
        ):
            measurements = [
                {"element_symbol": "Au", "value": au, "unit": "ppm"},
                {"element_symbol": "As", "value": arsenic_value, "unit": "ppm"},
                {"element_symbol": "Cu", "value": copper_value, "unit": "ppm"},
            ]
            if index == 4:
                measurements = [
                    row for row in measurements
                    if row["element_symbol"] != "Cu"
                ]
            samples.append({
                "sample_code": f"G{index}",
                "measurements": measurements,
            })

        result = calculate_element_associations(
            samples,
            preprocessing={"log_transform": True},
            association_options={
                "method": "spearman",
                "minimum_shared_samples": 5,
                "minimum_absolute_correlation": 0.7,
                "maximum_adjusted_p_value": 0.05,
            },
        )
        au_as = next(
            row for row in result["associations"]
            if {row["element_a"], row["element_b"]} == {"Au", "As"}
        )

        self.assertEqual(au_as["correlation"], 1.0)
        self.assertEqual(au_as["reliability"], "moderate")
        self.assertTrue(au_as["passes_filters"])

    def test_invalid_association_settings_are_rejected(self):
        with self.assertRaisesMessage(ValueError, "Pearson or Spearman"):
            resolve_association_options({"method": "made-up"})
        with self.assertRaisesMessage(ValueError, "at least three"):
            resolve_association_options({"minimum_shared_samples": 2})



class ElementAssociationPersistenceTests(TestCase):
    def test_completed_analysis_stores_the_association_block(self):
        samples = [
            sample("S1", 1, 2, 8),
            sample("S2", 2, 4, 6),
            sample("S3", 3, 6, 4),
            sample("S4", 4, 8, 2),
        ]
        analysis = FullAnalysis.objects.create(
            name="Association run",
            sample_data={"samples": samples},
            parameters={
                "top_n": 5,
                "batch_size": 250,
                "similarity_method": "log_difference_similarity",
                "preprocessing": {},
                "selected_elements": ["Au", "Cu", "Pb"],
                "reference_count": 0,
            },
            status=FullAnalysis.STATUS_PENDING,
        )

        FullAnalysisListCreateView().process_full_analysis(analysis.id)
        analysis.refresh_from_db()

        self.assertEqual(analysis.status, FullAnalysis.STATUS_COMPLETED)
        self.assertTrue(analysis.element_associations["available"])
        self.assertEqual(analysis.element_associations["sample_count"], 4)
