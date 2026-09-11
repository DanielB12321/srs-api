"""Tests for the reusable geochemical profiles saved with an analysis."""

from django.test import SimpleTestCase

from ..models import GeochemicalSignature
from ..signatures import build_analysis_signatures
from .test_persistence import PersistenceTestCase


def sample(code, values):
    """Build the small request shape accepted by the analysis endpoint."""
    return {
        "sample_code": code,
        "measurements": [
            {"element_symbol": symbol, "value": value, "unit": "ppm"}
            for symbol, value in values.items()
        ],
    }


class SignatureBuilderTests(SimpleTestCase):
    """Check that sample and analysis-wide vectors use one preprocessing path."""

    def test_one_profile_is_built_for_each_sample_and_one_composite(self):
        profiles = build_analysis_signatures([
            sample("S1", {"Cu": 10, "Zn": 100}),
            sample("S2", {"Cu": 30, "Zn": 300}),
        ])

        self.assertEqual(len(profiles), 3)
        composite = next(row for row in profiles if row["kind"] == "composite")
        self.assertEqual(composite["representative_method"], "median")
        self.assertEqual(composite["raw_vector"], {"Cu": 20.0, "Zn": 200.0})

    def test_selected_elements_are_applied_to_the_saved_vector(self):
        profiles = build_analysis_signatures(
            [sample("S1", {"Au": 1, "Cu": 10, "Zn": 100})],
            selected_elements=["cu", "zn"],
        )

        self.assertEqual(profiles[0]["elements"], ["Cu", "Zn"])
        self.assertEqual(set(profiles[0]["vector"]), {"Cu", "Zn"})

    def test_log_and_clr_scales_are_identified(self):
        logged = build_analysis_signatures(
            [sample("S1", {"Cu": 10, "Zn": 100})],
            {"log_transform": True},
        )[0]
        clr = build_analysis_signatures(
            [sample("S1", {"Cu": 10, "Zn": 100})],
            {"normalise": True},
        )[0]

        self.assertEqual(logged["value_space"], "log10_ppm")
        self.assertEqual(logged["vector"], {"Cu": 1.0, "Zn": 2.0})
        self.assertEqual(clr["value_space"], "clr")
        self.assertAlmostEqual(sum(clr["vector"].values()), 0.0, places=12)


class StoredSignatureTests(PersistenceTestCase):
    """Check persistence and the API contract for saved profiles."""

    def test_completed_analysis_stores_sample_and_composite_profiles(self):
        full_analysis = self.run_analysis()

        signatures = full_analysis.signatures.all()
        self.assertEqual(signatures.count(), 2)
        self.assertTrue(signatures.filter(
            kind=GeochemicalSignature.KIND_SAMPLE,
            sample_index=0,
            sample_code="S001",
        ).exists())
        self.assertTrue(signatures.filter(
            kind=GeochemicalSignature.KIND_COMPOSITE,
            sample_index=-1,
            representative_method="median",
        ).exists())

    def test_rerunning_replaces_signatures_instead_of_duplicating_them(self):
        full_analysis = self.run_analysis()

        self.view.process_full_analysis(full_analysis.id)

        self.assertEqual(full_analysis.signatures.count(), 2)

    def test_detail_includes_the_composite_profile(self):
        full_analysis = self.run_analysis()

        payload = self.client.get(
            f"/api/full-analysis/{full_analysis.id}/"
        ).json()

        self.assertEqual(payload["signature_count"], 2)
        self.assertEqual(payload["signature_profile"]["kind"], "composite")
        self.assertEqual(
            set(payload["signature_profile"]["vector"]),
            {"Au", "Cu", "Zn"},
        )

    def test_signature_endpoint_can_filter_profile_kind(self):
        full_analysis = self.run_analysis()

        payload = self.client.get(
            f"/api/full-analysis/{full_analysis.id}/signatures/?kind=sample"
        ).json()

        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["results"][0]["sample_code"], "S001")

    def test_old_analysis_without_profiles_remains_readable(self):
        analysis = self.run_analysis()
        analysis.signatures.all().delete()

        payload = self.client.get(f"/api/full-analysis/{analysis.id}/").json()

        self.assertIsNone(payload["signature_profile"])
        self.assertEqual(payload["signature_count"], 0)

    def test_invalid_signature_kind_is_rejected(self):
        full_analysis = self.run_analysis()

        response = self.client.get(
            f"/api/full-analysis/{full_analysis.id}/signatures/?kind=unknown"
        )

        self.assertEqual(response.status_code, 400)
