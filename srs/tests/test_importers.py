"""Tests for uploaded CSV dataset imports."""

import tempfile

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import SimpleTestCase, TestCase, override_settings

from ..importers import run_dataset_import
from ..importers.osnaca import _to_float
from ..models import Dataset, Element, SampleMeasurement


class DatasetImportTests(TestCase):
    def setUp(self):
        self.media_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.media_directory.cleanup)

        media_settings = override_settings(MEDIA_ROOT=self.media_directory.name)
        media_settings.enable()
        self.addCleanup(media_settings.disable)

    def create_dataset(self, contents, filename="samples.csv"):
        return Dataset.objects.create(
            name=filename,
            uploaded_file=SimpleUploadedFile(
                filename,
                contents.encode("utf-8"),
                content_type="text/csv",
            ),
        )

    def test_csv_rows_and_measurements_are_imported(self):
        dataset = self.create_dataset(
            "sample_id,latitude,longitude,Cu_ppm,Zn_ppm\n"
            "S1,-27.4,153.1,10,20\n"
            "S2,,,5,\n"
        )

        run_dataset_import(dataset.id)
        dataset.refresh_from_db()

        self.assertEqual(dataset.status, Dataset.STATUS_COMPLETED)
        self.assertEqual(dataset.row_count, 2)
        self.assertEqual(dataset.col_count, 5)
        self.assertEqual(dataset.null_count, 1)
        self.assertEqual(dataset.samples.count(), 2)
        self.assertEqual(Element.objects.count(), 2)
        self.assertEqual(SampleMeasurement.objects.count(), 4)
        self.assertEqual(
            list(Element.objects.order_by("symbol").values_list("symbol", flat=True)),
            ["Cu", "Zn"],
        )

    def test_non_finite_measurements_fail_the_import(self):
        for index, value in enumerate(("NaN", "Infinity", "-Infinity"), start=1):
            with self.subTest(value=value):
                dataset = self.create_dataset(
                    f"sample_id,Cu_ppm\nS{index},{value}\n",
                    filename=f"non-finite-{index}.csv",
                )

                with self.assertRaisesRegex(ValueError, "Non-finite"):
                    run_dataset_import(dataset.id)

                dataset.refresh_from_db()
                self.assertEqual(dataset.status, Dataset.STATUS_FAILED)
                self.assertIn("Non-finite", dataset.errors[0])
                self.assertEqual(dataset.samples.count(), 0)

    def test_a_utf8_bom_on_the_first_heading_is_supported(self):
        dataset = self.create_dataset("\ufeffsample_id,Cu_ppm\nS1,10\n")

        run_dataset_import(dataset.id)
        dataset.refresh_from_db()

        self.assertEqual(dataset.status, Dataset.STATUS_COMPLETED)
        self.assertEqual(dataset.samples.count(), 1)

    def test_two_columns_for_the_same_element_are_rejected_clearly(self):
        dataset = self.create_dataset(
            "sample_id,Cu_ppm,Cu_ppb\nS1,10,10000\n"
        )

        with self.assertRaisesRegex(ValueError, "one measurement column"):
            run_dataset_import(dataset.id)

        dataset.refresh_from_db()
        self.assertEqual(dataset.status, Dataset.STATUS_FAILED)

    def test_coordinates_outside_real_world_ranges_are_rejected(self):
        dataset = self.create_dataset(
            "sample_id,latitude,longitude,Cu_ppm\nS1,95,153,10\n"
        )

        with self.assertRaisesRegex(ValueError, "Latitude must be between"):
            run_dataset_import(dataset.id)

        dataset.refresh_from_db()
        self.assertEqual(dataset.status, Dataset.STATUS_FAILED)


class OsnacaValueTests(SimpleTestCase):
    def test_non_finite_spreadsheet_values_are_ignored(self):
        self.assertIsNone(_to_float(float("nan")))
        self.assertIsNone(_to_float(float("inf")))
        self.assertIsNone(_to_float(float("-inf")))
