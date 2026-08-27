"""Tests for SRS management-command error handling."""

from types import SimpleNamespace
from unittest.mock import patch

from django.core.management.base import CommandError
from django.test import SimpleTestCase

from ..management.commands.import_osnaca import Command
from ..models import ReferenceImport


class ImportOsnacaCommandTests(SimpleTestCase):
    @patch("srs.management.commands.import_osnaca.run_import")
    def test_failed_import_returns_a_command_error(self, run_import):
        run_import.return_value = SimpleNamespace(
            id=7,
            status=ReferenceImport.STATUS_FAILED,
            stats={},
            errors=[{"reason": "Bad workbook"}],
        )

        with self.assertRaisesRegex(CommandError, "Import #7 failed"):
            Command()._run(7)
