"""Tests for analysis-wide element relationships."""

from django.test import SimpleTestCase, TestCase

from ..api_views import FullAnalysisListCreateView
from ..associations import calculate_element_associations
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
