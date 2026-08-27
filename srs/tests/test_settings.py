"""Checks for settings that should fail safely in deployed environments."""

from unittest.mock import patch

from django.core.exceptions import ImproperlyConfigured
from django.test import SimpleTestCase

from srs_api.settings import _django_secret_key


class SecretKeySettingsTests(SimpleTestCase):
    def test_deployment_requires_a_secret_key(self):
        with patch.dict("os.environ", {"SECRET_KEY": ""}):
            with self.assertRaises(ImproperlyConfigured):
                _django_secret_key(deployed=True)

    def test_local_development_keeps_a_fallback(self):
        with patch.dict("os.environ", {"SECRET_KEY": ""}):
            self.assertTrue(_django_secret_key(deployed=False))
