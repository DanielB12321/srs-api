"""Tests for concentration-unit conversion."""

from django.test import SimpleTestCase

from ..services.units import concentration_to_ppm


class ConcentrationConversionTests(SimpleTestCase):
    def test_common_geochemical_units_convert_to_ppm(self):
        self.assertEqual(concentration_to_ppm(42.5, "ppb"), 0.0425)
        self.assertEqual(concentration_to_ppm(3.2, "pct"), 32000.0)
        self.assertEqual(concentration_to_ppm(55.1, "ppm"), 55.1)
        self.assertEqual(concentration_to_ppm(1.5, "g/t"), 1.5)

    def test_unknown_units_are_not_silently_compared(self):
        self.assertIsNone(concentration_to_ppm(10, "mol/L"))

    def test_non_finite_values_are_rejected(self):
        self.assertIsNone(concentration_to_ppm(float("nan"), "ppm"))
        self.assertIsNone(concentration_to_ppm(float("inf"), "ppm"))
        self.assertIsNone(concentration_to_ppm(float("-inf"), "ppm"))

    def test_conversion_overflow_is_rejected(self):
        self.assertIsNone(concentration_to_ppm(1e308, "pct"))
