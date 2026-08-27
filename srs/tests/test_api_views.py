"""Regression tests for request validation and saved analysis data."""

import hashlib
import json
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings

from ..api_views import FullAnalysisListCreateView
from ..models import Dataset, FullAnalysis
from ..preprocessing import resolve_options


API_KEY = "test-shared-key"
KEY_HEADER = {"HTTP_X_SRS_API_KEY": API_KEY}


@override_settings(SRS_API_SHARED_KEY=API_KEY)
class APIViewRegressionTests(TestCase):
    def setUp(self):
        self.media_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.media_directory.cleanup)
        self.media_settings = override_settings(MEDIA_ROOT=self.media_directory.name)
        self.media_settings.enable()
        self.addCleanup(self.media_settings.disable)

    def sample_payload(self, below_detection_limit=False):
        return {
            "analysis_name": "Request test",
            "samples": [{
                "sample_code": "S1",
                "measurements": [{
                    "element_symbol": "Cu",
                    "value": 10,
                    "unit": "ppm",
                    "below_detection_limit": below_detection_limit,
                }],
            }],
        }

    def test_search_rejects_a_non_numeric_measurement_limit(self):
        response = self.client.get(
            "/api/reference-library/search/?element=Cu&min_value=not-a-number",
            **KEY_HEADER,
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json()["error"],
            "min_value and max_value must be numbers.",
        )

    @patch("srs.api_views.threading.Thread")
    @override_settings(SRS_DEFAULT_ALGORITHM="correlation")
    def test_analysis_uses_the_configured_default_algorithm(self, thread):
        response = self.client.post(
            "/api/full-analysis/",
            self.sample_payload(),
            content_type="application/json",
            **KEY_HEADER,
        )

        self.assertEqual(response.status_code, 202)
        analysis = FullAnalysis.objects.get(id=response.json()["full_analysis_id"])
        self.assertEqual(analysis.method, "correlation")
        self.assertEqual(
            analysis.parameters["algorithm_id"],
            "correlation",
        )
        thread.return_value.start.assert_called_once_with()

    @patch("srs.api_views.threading.Thread")
    def test_string_false_is_saved_as_false(self, thread):
        response = self.client.post(
            "/api/full-analysis/",
            self.sample_payload(below_detection_limit="false"),
            content_type="application/json",
            **KEY_HEADER,
        )

        self.assertEqual(response.status_code, 202)
        analysis = FullAnalysis.objects.get(id=response.json()["full_analysis_id"])
        measurement = analysis.sample_data["samples"][0]["measurements"][0]
        self.assertIs(measurement["below_detection_limit"], False)
        thread.return_value.start.assert_called_once_with()

    @patch("srs.api_views.threading.Thread")
    def test_analysis_saves_detail_and_neighbour_limits(self, thread):
        payload = self.sample_payload()
        payload.update({"detail_top_n": 4, "k": 3})

        response = self.client.post(
            "/api/full-analysis/",
            payload,
            content_type="application/json",
            **KEY_HEADER,
        )

        self.assertEqual(response.status_code, 202)
        analysis = FullAnalysis.objects.get(id=response.json()["full_analysis_id"])
        self.assertEqual(analysis.parameters["detail_top_n"], 4)
        self.assertEqual(analysis.parameters["k"], 3)
        thread.return_value.start.assert_called_once_with()

    def test_analysis_rejects_invalid_detail_limits(self):
        payload = self.sample_payload()
        payload["k"] = "many"

        response = self.client.post(
            "/api/full-analysis/",
            payload,
            content_type="application/json",
            **KEY_HEADER,
        )

        self.assertEqual(response.status_code, 400)

    @patch("srs.api_views.threading.Thread")
    @override_settings(SRS_DEFAULT_ALGORITHM="log_difference_similarity")
    def test_unknown_algorithm_is_saved_as_the_default(self, thread):
        payload = self.sample_payload()
        payload["similarity_method"] = "not-a-real-algorithm"

        response = self.client.post(
            "/api/full-analysis/",
            payload,
            content_type="application/json",
            **KEY_HEADER,
        )

        self.assertEqual(response.status_code, 202)
        analysis = FullAnalysis.objects.get(id=response.json()["full_analysis_id"])
        self.assertEqual(analysis.method, "log_difference_similarity")
        self.assertEqual(
            analysis.parameters["requested_similarity_method"],
            "not-a-real-algorithm",
        )
        thread.return_value.start.assert_called_once_with()

    @patch("srs.api_views.run_dataset_import")
    def test_a_failed_upload_does_not_block_a_clean_retry(self, run_import):
        contents = b"sample_id,Cu_ppm\nS1,10\n"
        failed = Dataset.objects.create(
            name="Failed upload",
            file_sha256=hashlib.sha256(contents).hexdigest(),
            status=Dataset.STATUS_FAILED,
        )
        upload = SimpleUploadedFile("samples.csv", contents, content_type="text/csv")

        response = self.client.post(
            "/api/datasets/",
            {"name": "Retry", "uploaded_file": upload},
            **KEY_HEADER,
        )

        self.assertEqual(response.status_code, 201)
        self.assertNotEqual(response.json()["id"], failed.id)
        run_import.assert_called_once()

    def test_analysis_detail_accepts_legacy_string_sample_data(self):
        analysis = FullAnalysis.objects.create(
            name="Legacy",
            uploaded_sample_code="OLD",
            sample_data=json.dumps({
                "samples": [{
                    "sample_code": "OLD",
                    "name": "Old sample",
                    "measurements": [],
                }],
            }),
        )

        response = self.client.get(
            f"/api/full-analysis/{analysis.id}/",
            **KEY_HEADER,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["analysed_samples"][0]["sample_code"], "OLD")

    def test_whole_library_path_uses_the_current_sample_code(self):
        analysis = FullAnalysis.objects.create(
            name="Two samples",
            uploaded_sample_code="S1",
            sample_data={
                "samples": [
                    {"sample_code": "S1", "measurements": []},
                    {"sample_code": "S2", "measurements": []},
                ],
            },
        )
        measurements = [{
            "element_symbol": "Cu",
            "value": 10,
            "unit": "ppm",
        }]
        algorithm = Mock()
        algorithm.compare.return_value = SimpleNamespace(matches=[])

        FullAnalysisListCreateView().create_ranked_matches_via_compare(
            algorithm,
            analysis,
            measurements,
            sample_index=1,
            top_n=10,
            batch_size=50,
            options=resolve_options({}, None),
        )

        compared_samples = algorithm.compare.call_args.args[0]
        self.assertEqual(compared_samples[0]["sample_code"], "S2")
