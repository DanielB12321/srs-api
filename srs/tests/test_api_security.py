"""Checks for the server-to-server API key and optional audit headers."""

import tempfile
from unittest.mock import patch

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings

from ..models import Dataset, Element, FullAnalysis, Sample, SampleMeasurement


SHARED_KEY = "test-shared-key"
KEY_HEADER = {"HTTP_X_SRS_API_KEY": SHARED_KEY}


@override_settings(SRS_API_SHARED_KEY=SHARED_KEY)
class SharedAPIKeyTests(TestCase):
    def setUp(self):
        # FileField writes are kept outside the repository during upload tests.
        self.media_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.media_directory.cleanup)
        self.media_settings = override_settings(
            MEDIA_ROOT=self.media_directory.name,
        )
        self.media_settings.enable()
        self.addCleanup(self.media_settings.disable)

    def test_functional_endpoint_rejects_a_missing_key(self):
        response = self.client.get("/api/algorithms/")

        self.assertEqual(response.status_code, 401)

    def test_functional_endpoint_rejects_the_wrong_key(self):
        response = self.client.get(
            "/api/algorithms/",
            HTTP_X_SRS_API_KEY="wrong-key",
        )

        self.assertEqual(response.status_code, 401)

    @override_settings(SRS_API_SHARED_KEY="")
    def test_an_unset_server_key_fails_closed(self):
        response = self.client.get("/api/algorithms/", **KEY_HEADER)

        self.assertEqual(response.status_code, 401)

    def test_the_correct_key_allows_a_request_without_user_headers(self):
        response = self.client.get("/api/algorithms/", **KEY_HEADER)

        self.assertEqual(response.status_code, 200)

    def test_malformed_optional_user_details_are_rejected(self):
        response = self.client.get(
            "/api/algorithms/",
            **KEY_HEADER,
            HTTP_X_SRS_USER_ID="someone",
        )

        self.assertEqual(response.status_code, 401)

    def test_schema_and_docs_stay_public(self):
        schema = self.client.get("/api/schema/")
        docs = self.client.get("/api/docs/")

        self.assertEqual(schema.status_code, 200)
        self.assertEqual(docs.status_code, 200)
        self.assertIn(b"SRSApiKey", schema.content)
        self.assertIn(b"X-SRS-API-Key", schema.content)

    def test_all_datasets_remain_shared_including_unowned_records(self):
        Dataset.objects.create(name="Old unowned dataset")
        Dataset.objects.create(
            name="Another user's dataset",
            uploaded_by_id=222,
            uploaded_by_email="other@example.com",
        )

        response = self.client.get(
            "/api/datasets/",
            **KEY_HEADER,
            HTTP_X_SRS_USER_ID="111",
            HTTP_X_SRS_USER_EMAIL="current@example.com",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["count"], 2)

    def test_shared_records_can_be_changed_by_a_different_audit_user(self):
        dataset = Dataset.objects.create(
            name="Created by another user",
            uploaded_by_id=222,
        )

        changed = self.client.patch(
            f"/api/datasets/{dataset.id}/",
            {"name": "Changed by current user"},
            content_type="application/json",
            **KEY_HEADER,
            HTTP_X_SRS_USER_ID="111",
        )
        removed = self.client.delete(
            f"/api/datasets/{dataset.id}/",
            **KEY_HEADER,
            HTTP_X_SRS_USER_ID="111",
        )

        self.assertEqual(changed.status_code, 200)
        self.assertEqual(changed.json()["name"], "Changed by current user")
        self.assertEqual(removed.status_code, 204)
        self.assertFalse(Dataset.objects.filter(id=dataset.id).exists())

    def test_samples_and_measurements_remain_shared(self):
        dataset = Dataset.objects.create(name="Shared", uploaded_by_id=222)
        sample = Sample.objects.create(dataset=dataset, sample_code="S1")
        element = Element.objects.create(symbol="Cu")
        measurement = SampleMeasurement.objects.create(
            sample=sample,
            element=element,
            value=12.5,
            unit="ppm",
        )

        sample_response = self.client.get(
            f"/api/samples/{sample.id}/",
            **KEY_HEADER,
            HTTP_X_SRS_USER_ID="111",
        )
        measurement_response = self.client.get(
            f"/api/sample-measurements/{measurement.id}/",
            **KEY_HEADER,
            HTTP_X_SRS_USER_ID="111",
        )

        self.assertEqual(sample_response.status_code, 200)
        self.assertEqual(measurement_response.status_code, 200)

    @patch("srs.api_views.run_dataset_import")
    def test_dataset_creation_uses_headers_for_audit_not_body_values(
        self,
        run_dataset_import,
    ):
        uploaded_file = SimpleUploadedFile(
            "samples.csv",
            b"sample_id,Cu_ppm\nS1,10\n",
            content_type="text/csv",
        )

        response = self.client.post(
            "/api/datasets/",
            {
                "name": "Audit test",
                "uploaded_file": uploaded_file,
                "uploaded_by_id": 999,
                "uploaded_by_email": "spoofed@example.com",
            },
            **KEY_HEADER,
            HTTP_X_SRS_USER_ID="123",
            HTTP_X_SRS_USER_EMAIL="real@example.com",
        )

        self.assertEqual(response.status_code, 201)
        dataset = Dataset.objects.get(id=response.json()["id"])
        self.assertEqual(dataset.uploaded_by_id, 123)
        self.assertEqual(dataset.uploaded_by_email, "real@example.com")
        run_dataset_import.assert_called_once_with(dataset.id)

    @patch("srs.api_views.threading.Thread")
    def test_full_analysis_creation_saves_optional_audit_details(self, thread):
        response = self.client.post(
            "/api/full-analysis/",
            {
                "analysis_name": "Audit analysis",
                "samples": [{
                    "sample_code": "S1",
                    "measurements": [{
                        "element_symbol": "Cu",
                        "value": 10,
                        "unit": "ppm",
                    }],
                }],
            },
            content_type="application/json",
            **KEY_HEADER,
            HTTP_X_SRS_USER_ID="123",
            HTTP_X_SRS_USER_EMAIL="real@example.com",
        )

        self.assertEqual(response.status_code, 202)
        analysis = FullAnalysis.objects.get(
            id=response.json()["full_analysis_id"]
        )
        self.assertEqual(analysis.created_by_id, 123)
        self.assertEqual(analysis.created_by_email, "real@example.com")
        thread.return_value.start.assert_called_once_with()

    def test_analyses_are_not_filtered_by_the_audit_user(self):
        FullAnalysis.objects.create(name="Unowned")
        FullAnalysis.objects.create(name="Other", created_by_id=222)

        response = self.client.get(
            "/api/full-analysis/",
            **KEY_HEADER,
            HTTP_X_SRS_USER_ID="111",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["count"], 2)
