"""Regression tests for OSNACA measurement units."""

from importlib import import_module

from django.apps import apps
from django.test import TestCase
from openpyxl import Workbook

from ..importers.osnaca import load_measurements
from ..models import (
    Element,
    ReferenceImport,
    ReferenceSample,
    ReferenceSampleMeasurement,
)


class OsnacaMeasurementImportTests(TestCase):
    def setUp(self):
        self.import_row = ReferenceImport.objects.create(
            source_name="OSNACA",
            data_file="srs/reference_imports/data.xlsx",
            metadata_file="srs/reference_imports/metadata.xlsx",
        )
        self.sample = ReferenceSample.objects.create(
            import_ref=self.import_row,
            sample_code="S1",
            source_dataset="OSNACA",
        )
        self.censored_sample = ReferenceSample.objects.create(
            import_ref=self.import_row,
            sample_code="S2",
            source_dataset="OSNACA",
        )
        Element.objects.create(
            symbol="Fe",
            name="Iron",
            atomic_number=26,
            default_unit="%",
        )

    def workbook(self, measured_value):
        workbook = Workbook()
        worksheet = workbook.active
        worksheet.title = "Data"
        worksheet.append(["Sample", "Fe"])
        worksheet.append(["Sample", "Fe"])
        worksheet.append(["S1", measured_value])
        worksheet.append(["S2", -500])
        return workbook

    def import_measurements(self, measured_value):
        workbook = self.workbook(measured_value)
        self.addCleanup(workbook.close)
        stats = {}
        errors = []
        load_measurements(
            workbook,
            self.import_row,
            {
                self.sample.sample_code: self.sample,
                self.censored_sample.sample_code: self.censored_sample,
            },
            stats,
            errors,
        )
        self.assertEqual(errors, [])

    def test_major_elements_are_converted_from_ppm_to_percent(self):
        self.import_measurements(82500)

        measurement = ReferenceSampleMeasurement.objects.get(
            reference_sample=self.sample,
            element__symbol="Fe",
        )
        self.assertEqual(measurement.value, 8.25)
        self.assertEqual(measurement.unit, "%")

        censored = ReferenceSampleMeasurement.objects.get(
            reference_sample=self.censored_sample,
            element__symbol="Fe",
        )
        self.assertIsNone(censored.value)
        self.assertTrue(censored.below_detection_limit)
        self.assertEqual(censored.detection_limit, 0.05)
        self.assertEqual(censored.unit, "%")

    def test_rerun_updates_existing_measurements(self):
        self.import_measurements(82500)
        self.import_measurements(65000)

        measurements = ReferenceSampleMeasurement.objects.filter(
            reference_sample=self.sample,
            element__symbol="Fe",
        )
        self.assertEqual(measurements.count(), 1)
        self.assertEqual(measurements.get().value, 6.5)

    def test_data_migration_corrects_existing_percent_measurements(self):
        measurement = ReferenceSampleMeasurement.objects.create(
            import_ref=self.import_row,
            reference_sample=self.sample,
            element=Element.objects.get(symbol="Fe"),
            value=82500,
            unit="%",
            below_detection_limit=False,
            detection_limit=500,
        )
        migration = import_module(
            "srs.migrations.0013_correct_osnaca_percent_measurements"
        )

        migration.correct_osnaca_percent_values(apps, None)

        measurement.refresh_from_db()
        self.assertEqual(measurement.value, 8.25)
        self.assertEqual(measurement.detection_limit, 0.05)

        migration.restore_raw_osnaca_values(apps, None)
        measurement.refresh_from_db()
        self.assertEqual(measurement.value, 82500)
        self.assertEqual(measurement.detection_limit, 500)
