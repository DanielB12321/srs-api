"""Tests for SRS management-command error handling."""

from types import SimpleNamespace
from unittest.mock import patch

from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import SimpleTestCase

from ..management.commands.import_osnaca import Command
from ..models import ReferenceImport


class BenchmarkAlgorithmsCommandTests(SimpleTestCase):
    @patch("srs.management.commands.benchmark_algorithms.run_benchmark")
    @patch("srs.management.commands.benchmark_algorithms.load_signatures")
    def test_normalise_flag_reaches_the_preprocessing_block(
        self, load_signatures, run_benchmark
    ):
        load_signatures.return_value = [
            {"deposit_pk": 1, "deposit_class": "VHMS"},
        ]
        run_benchmark.return_value = SimpleNamespace(
            algorithm_id="correlation", algorithm_version="1.0.0",
            protocol="deposit", n_queries=1, top_1=1.0, top_5=1.0,
            mean_reciprocal_rank=1.0, majority_baseline=0.5,
            lift_over_baseline=0.5, runtime_ms=1.0, ms_per_query=1.0,
            per_class={},
        )

        call_command(
            "benchmark_algorithms", "--algorithms", "correlation", "--normalise",
        )

        expected = {"handle_missing": "half_dl", "normalise": True}
        self.assertEqual(load_signatures.call_args.args[0], expected)
        self.assertEqual(
            run_benchmark.call_args.kwargs["preprocessing"], expected
        )


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
