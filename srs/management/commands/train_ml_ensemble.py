"""Train and save the optional ML ensemble model files."""

from __future__ import annotations

import json
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from ...training.ml_ensemble_training import train_from_file


class Command(BaseCommand):
    help = (
        "Train the XGBoost and RBF-SVM ensemble and save its model files."
    )

    def add_arguments(self, parser):
        parser.add_argument("--input", required=True, help="OSNACA CSV/XLSX path.")
        parser.add_argument(
            "--sheet",
            default=None,
            help="Excel sheet name, e.g. 'Data 24 clip'.",
        )
        parser.add_argument(
            "--preprocessing-json",
            default=None,
            help=(
                "Optional JSON file containing SRS preprocessing settings. "
                "Omit it to use the defaults."
            ),
        )
        parser.add_argument(
            "--output",
            default=None,
            help=(
                "Artifact output directory. Defaults to "
                "srs/models/ml_ensemble."
            ),
        )

    def handle(self, *args, **options):
        input_path = Path(options["input"]).resolve()

        if options["output"]:
            output_directory = Path(options["output"]).resolve()
        else:
            output_directory = (
                Path(__file__).resolve().parents[2]
                / "models"
                / "ml_ensemble"
            )

        preprocessing_request = None
        preprocessing_path = options.get("preprocessing_json")
        if preprocessing_path:
            try:
                with Path(preprocessing_path).open("r", encoding="utf-8") as file:
                    preprocessing_request = json.load(file)
            except (OSError, json.JSONDecodeError) as exc:
                raise CommandError(
                    f"Could not read preprocessing JSON: {exc}"
                ) from exc

        try:
            manifest = train_from_file(
                input_path=input_path,
                output_directory=output_directory,
                sheet_name=options.get("sheet"),
                preprocessing_request=preprocessing_request,
            )
        except Exception as exc:
            raise CommandError(str(exc)) from exc

        self.stdout.write(self.style.SUCCESS("ML ensemble training complete."))
        self.stdout.write(f"Artifacts: {output_directory}")
        self.stdout.write(
            "Final-test metrics: "
            + json.dumps(manifest["final_test_metrics"], indent=2)
        )
