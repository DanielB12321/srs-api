"""
Tests for persisting and exposing the envelope.

The load-bearing test here is the backwards-compatibility one. Everything else
added in this phase is additive, and the way that claim fails in practice is
that a new key appears in a response the frontend already parses. So the shape
of the existing responses is asserted directly, key by key.
"""

from django.test import TestCase, override_settings

from ..api_views import FullAnalysisListCreateView, reference_library_version
from ..models import (
    Element,
    FullAnalysis,
    FullAnalysisMatch,
    ReferenceDeposit,
    ReferenceImport,
    ReferenceSample,
    ReferenceSampleMeasurement,
)
from ..preprocessing import PIPELINE_VERSION
from .test_characterisation import INPUT_VALUES, REFERENCE_VALUES

# The exact keys the separate frontend repository reads today. A new key here
# is fine; a missing or renamed one breaks a page.
SUMMARY_KEYS = {
    "id", "name", "uploaded_sample_code", "source_filename", "method", "status",
    "created_at", "completed_at", "match_count", "sample_count", "sample_codes",
    "selected_elements", "preprocessing", "parameters",
}
MATCH_KEYS = {"id", "rank", "similarity_score"}


@override_settings(SRS_API_SHARED_KEY="test-shared-key")
class PersistenceTestCase(TestCase):
    """Shared fixture: a small library with deposits, so confidence has data."""

    def setUp(self):
        self.client.defaults["HTTP_X_SRS_API_KEY"] = "test-shared-key"
        self.view = FullAnalysisListCreateView()
        self.elements = {
            symbol: Element.objects.create(symbol=symbol)
            for symbol in ("Cu", "Zn", "Au")
        }
        self.import_row = ReferenceImport.objects.create(
            source_name="OSNACA v1",
            status=ReferenceImport.STATUS_COMPLETED,
        )
        self.woodlawn = ReferenceDeposit.objects.create(
            name="Woodlawn", three_char_code="WLN", deposit_type="VHMS",
        )
        self.other = ReferenceDeposit.objects.create(
            name="Other", three_char_code="OTH", deposit_type="Orogenic Au",
        )
        self.add_reference("R1", REFERENCE_VALUES["identical"], self.woodlawn)
        self.add_reference("R2", REFERENCE_VALUES["cu_x10"], self.woodlawn)
        self.add_reference("R3", REFERENCE_VALUES["swapped"], self.other)

    def add_reference(self, sample_code, values, deposit):
        reference_sample = ReferenceSample.objects.create(
            import_ref=self.import_row,
            reference_deposit=deposit,
            sample_code=sample_code,
        )
        for symbol, value in values.items():
            ReferenceSampleMeasurement.objects.create(
                reference_sample=reference_sample,
                element=self.elements[symbol],
                value=value,
                unit="ppm",
            )
        return reference_sample

    def run_analysis(self, similarity_method="log_difference_similarity", **parameters):
        full_analysis = FullAnalysis.objects.create(
            name="Persisted analysis",
            uploaded_sample_code="S001",
            sample_data={"samples": [{
                "sample_code": "S001",
                "name": "S001",
                "latitude": -35.1,
                "longitude": 149.6,
                "measurements": [
                    {
                        "element_symbol": symbol,
                        "value": value,
                        "unit": "ppm",
                        "below_detection_limit": False,
                        "detection_limit": None,
                    }
                    for symbol, value in INPUT_VALUES.items()
                ],
            }]},
            method=similarity_method,
            parameters={
                "top_n": 200,
                "batch_size": 250,
                "similarity_method": similarity_method,
                "preprocessing": {},
                "selected_elements": [],
                "reference_count": 3,
                **parameters,
            },
        )
        self.view.process_full_analysis(full_analysis.id)
        full_analysis.refresh_from_db()
        return full_analysis


class RunProvenanceTests(PersistenceTestCase):
    """What was recorded about the run itself."""

    def test_provenance_is_written_as_real_columns(self):
        full_analysis = self.run_analysis("knn_aitchison")

        self.assertEqual(full_analysis.algorithm_id, "knn_aitchison")
        self.assertEqual(full_analysis.algorithm_version, "1.0.0")
        self.assertEqual(full_analysis.pipeline_version, PIPELINE_VERSION)
        self.assertEqual(full_analysis.reference_library_version, "1:OSNACA v1")

    def test_runtime_is_recorded(self):
        full_analysis = self.run_analysis()

        self.assertIsNotNone(full_analysis.runtime_ms)
        self.assertGreaterEqual(full_analysis.runtime_ms, 0.0)

    def test_an_unknown_method_records_what_actually_ran(self):
        """
        The request asked for something that does not exist, so the default
        ran. Recording the requested name would make the stored run a lie.
        """
        full_analysis = self.run_analysis("not_a_real_method")

        self.assertEqual(full_analysis.algorithm_id, "log_difference_similarity")

    def test_provenance_columns_are_queryable(self):
        """The reason these are columns rather than keys in the JSON blob."""
        self.run_analysis("knn_aitchison")
        self.run_analysis("log_difference_similarity")

        self.assertEqual(
            FullAnalysis.objects.filter(algorithm_id="knn_aitchison").count(),
            1,
        )

    def test_the_library_version_falls_back_when_nothing_is_imported(self):
        ReferenceImport.objects.all().delete()

        self.assertEqual(reference_library_version(), "")

    def test_the_library_version_tracks_the_most_recent_import(self):
        ReferenceImport.objects.create(
            source_name="OSNACA v2",
            status=ReferenceImport.STATUS_COMPLETED,
        )

        self.assertEqual(reference_library_version(), "2:OSNACA v2")

    def test_an_incomplete_import_is_not_used_as_the_version(self):
        ReferenceImport.objects.create(
            source_name="OSNACA v3 half done",
            status=ReferenceImport.STATUS_RUNNING,
        )

        self.assertEqual(reference_library_version(), "1:OSNACA v1")

    def test_sample_results_summarise_the_leading_matches(self):
        full_analysis = self.run_analysis()

        self.assertEqual(len(full_analysis.sample_results), 1)
        summary = full_analysis.sample_results[0]
        self.assertEqual(summary["sample_id"], "S001")
        self.assertEqual(summary["lat"], -35.1)
        self.assertEqual(summary["top_matches"][0]["deposit_id"], "WLN")


class MatchDetailTests(PersistenceTestCase):
    """The per-match blocks, and the size limit that governs them."""

    def matches(self, full_analysis):
        return list(
            full_analysis.ranked_matches
            .filter(analysed_sample_index=0)
            .order_by("rank")
        )

    def test_an_algorithm_with_no_extra_output_stores_nothing_extra(self):
        """
        The concentration-based methods offer no raw metrics, confidence, or
        evidence, so their rows stay exactly as compact as they were.
        """
        matches = self.matches(self.run_analysis("log_difference_similarity"))

        for match in matches:
            with self.subTest(rank=match.rank):
                self.assertIsNone(match.scores)
                self.assertIsNone(match.confidence)
                self.assertIsNone(match.evidence)

    def test_knn_stores_its_raw_metrics_confidence_and_evidence(self):
        matches = self.matches(self.run_analysis("knn_aitchison"))
        top = matches[0]

        self.assertIn("aitchison_distance", top.scores)
        self.assertIn("spearman_rho", top.scores)
        self.assertIn("consistency", top.confidence)
        self.assertIn("supporting", top.evidence)
        self.assertIn("conflicting", top.evidence)

    def test_evidence_entries_carry_the_contracted_fields(self):
        top = self.matches(self.run_analysis("knn_aitchison"))[0]
        entries = top.evidence["supporting"] + top.evidence["conflicting"]

        self.assertTrue(entries)
        for entry in entries:
            self.assertEqual(set(entry), {"element", "contribution", "imputed"})

    def test_detail_is_limited_to_the_configured_number_of_matches(self):
        """
        The size decision, enforced. Evidence is about 3.4 KB per row, and a
        real analysis ranks hundreds of references per sample.
        """
        matches = self.matches(
            self.run_analysis("knn_aitchison", detail_top_n=2)
        )

        self.assertEqual(len(matches), 3)
        self.assertIsNotNone(matches[0].evidence)
        self.assertIsNotNone(matches[1].evidence)
        self.assertIsNone(matches[2].evidence)

    def test_ranking_is_unaffected_by_how_much_detail_is_stored(self):
        """Detail is attached after ranking, so it cannot influence it."""
        full = self.matches(self.run_analysis("knn_aitchison", detail_top_n=10))
        limited = self.matches(self.run_analysis("knn_aitchison", detail_top_n=1))

        self.assertEqual(
            [(m.rank, m.reference_sample_id, m.similarity_score) for m in full],
            [(m.rank, m.reference_sample_id, m.similarity_score) for m in limited],
        )

    def test_confidence_reflects_the_deposits_of_the_nearest_references(self):
        """Two of the three references are Woodlawn, and both rank above Other."""
        matches = self.matches(self.run_analysis("knn_aitchison"))
        by_deposit = {
            match.reference_sample.reference_deposit.three_char_code: match
            for match in matches
        }

        self.assertEqual(by_deposit["WLN"].confidence["n_reference_samples"], 2)
        self.assertEqual(by_deposit["OTH"].confidence["n_reference_samples"], 1)


class BackwardsCompatibilityTests(PersistenceTestCase):
    """
    The hard constraint: the separate frontend repository must keep working.

    Every key it reads is asserted present, and the pre-existing match keys are
    asserted to be exactly what they were for an algorithm that adds nothing.
    """

    def test_the_analysis_list_keeps_every_key_it_had(self):
        self.run_analysis()

        payload = self.client.get("/api/full-analysis/").json()

        self.assertIn("count", payload)
        self.assertTrue(SUMMARY_KEYS.issubset(payload["results"][0]))

    def test_the_analysis_detail_keeps_every_key_it_had(self):
        full_analysis = self.run_analysis()

        payload = self.client.get(f"/api/full-analysis/{full_analysis.id}/").json()

        self.assertEqual(payload["full_analysis_id"], full_analysis.id)
        self.assertTrue({
            "id", "name", "uploaded_sample_code", "source_filename", "method",
            "parameters", "status", "created_at", "completed_at",
        }.issubset(payload["full_analysis"]))
        self.assertTrue({
            "sample_index", "sample_code", "name", "latitude", "longitude",
            "match_count",
        }.issubset(payload["analysed_samples"][0]))

    def test_a_ranked_match_is_unchanged_for_an_algorithm_adding_nothing(self):
        """Byte-for-byte the old shape, not merely a superset of it."""
        full_analysis = self.run_analysis("log_difference_similarity")

        payload = self.client.get(
            f"/api/full-analysis/{full_analysis.id}/samples/0/"
        ).json()

        for match in payload["ranked_matches"]["results"]:
            self.assertEqual(set(match), MATCH_KEYS)

    def test_new_blocks_appear_only_when_an_algorithm_produced_them(self):
        full_analysis = self.run_analysis("knn_aitchison", detail_top_n=1)

        results = self.client.get(
            f"/api/full-analysis/{full_analysis.id}/samples/0/"
        ).json()["ranked_matches"]["results"]

        self.assertTrue(MATCH_KEYS.issubset(results[0]))
        self.assertIn("evidence", results[0])
        # Beyond the detail cutoff the shape is exactly the original one.
        self.assertEqual(set(results[1]), MATCH_KEYS)

    def test_the_paginated_match_envelope_keeps_its_shape(self):
        full_analysis = self.run_analysis()

        ranked = self.client.get(
            f"/api/full-analysis/{full_analysis.id}/samples/0/?page=1&page_size=2"
        ).json()["ranked_matches"]

        self.assertEqual(
            set(ranked),
            {"count", "page", "page_size", "total_pages", "results"},
        )
        self.assertEqual(ranked["count"], 3)
        self.assertEqual(ranked["page_size"], 2)
        self.assertEqual(ranked["total_pages"], 2)

    def test_the_map_endpoint_keeps_its_shape(self):
        full_analysis = self.run_analysis()

        payload = self.client.get(
            f"/api/full-analysis/{full_analysis.id}/samples/0/map/"
        ).json()

        self.assertEqual(
            set(payload),
            {"sample_index", "sample_code", "count", "results"},
        )

    def test_an_analysis_saved_before_this_phase_still_serialises(self):
        """
        Rows that predate the migration have empty provenance and null detail.
        Reading one must not fail.
        """
        legacy = FullAnalysis.objects.create(
            name="Legacy", uploaded_sample_code="OLD",
            sample_data={"samples": [{"sample_code": "OLD", "measurements": []}]},
            parameters={"top_n": 200},
        )
        FullAnalysisMatch.objects.create(
            full_analysis=legacy,
            reference_sample=ReferenceSample.objects.first(),
            analysed_sample_index=0, rank=1, similarity_score=0.5,
        )

        detail = self.client.get(f"/api/full-analysis/{legacy.id}/").json()
        sample = self.client.get(
            f"/api/full-analysis/{legacy.id}/samples/0/"
        ).json()

        self.assertEqual(detail["full_analysis"]["algorithm_id"], "")
        self.assertIsNone(detail["full_analysis"]["runtime_ms"])
        self.assertIsNone(detail["sample_results"])
        self.assertEqual(detail["full_analysis"]["status"], "completed")
        self.assertEqual(set(sample["ranked_matches"]["results"][0]), MATCH_KEYS)
