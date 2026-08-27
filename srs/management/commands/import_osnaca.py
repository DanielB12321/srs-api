"""Import an OSNACA workbook pair from the command line."""

import hashlib
from pathlib import Path

from django.core.files import File
from django.core.management.base import BaseCommand, CommandError

from ...importers import run_import
from ...models import ReferenceImport


class Command(BaseCommand):
    help = "Import an OSNACA workbook pair, or re-run an existing import."

    def add_arguments(self, parser):
        parser.add_argument("--data", type=Path, help="Path to OSNACA-Data*.xlsx")
        parser.add_argument("--metadata", type=Path, help="Path to OSNACA-Metadata*.xlsx")
        parser.add_argument("--source-name", default="OSNACA")
        parser.add_argument("--description", default="")
        parser.add_argument(
            "--rerun",
            type=int,
            default=None,
            help="ReferenceImport id to re-run; skips file upload.",
        )

    def handle(self, *args, **options):
        if options["rerun"]:
            self._rerun(options["rerun"])
        else:
            self._fresh(options)

    def _fresh(self, options):
        data_path: Path = options["data"]
        meta_path: Path = options["metadata"]
        if not data_path or not meta_path:
            raise CommandError("--data and --metadata are required (or use --rerun).")
        if not data_path.is_file():
            raise CommandError(f"Data file not found: {data_path}")
        if not meta_path.is_file():
            raise CommandError(f"Metadata file not found: {meta_path}")

        data_hash = _sha256_of_file(data_path)
        meta_hash = _sha256_of_file(meta_path)

        existing = ReferenceImport.objects.filter(
            data_sha256=data_hash, metadata_sha256=meta_hash,
        ).first()
        if existing:
            self.stdout.write(self.style.WARNING(
                f"Identical files already imported as #{existing.id} "
                f"({existing.status}). Re-running it instead."
            ))
            self._run(existing.id)
            return

        with data_path.open("rb") as df, meta_path.open("rb") as mf:
            import_row = ReferenceImport.objects.create(
                source_name=options["source_name"],
                description=options["description"],
                data_file=File(df, name=data_path.name),
                metadata_file=File(mf, name=meta_path.name),
                data_sha256=data_hash,
                metadata_sha256=meta_hash,
                status=ReferenceImport.STATUS_PENDING,
            )

        self.stdout.write(f"Created ReferenceImport #{import_row.id}; running...")
        self._run(import_row.id)

    def _rerun(self, import_id: int) -> None:
        try:
            ReferenceImport.objects.get(pk=import_id)
        except ReferenceImport.DoesNotExist:
            raise CommandError(f"ReferenceImport {import_id} not found.")
        self.stdout.write(f"Re-running ReferenceImport #{import_id}...")
        self._run(import_id)

    def _run(self, import_id: int) -> None:
        import_row = run_import(import_id)
        if import_row.status != ReferenceImport.STATUS_COMPLETED:
            error_count = len(import_row.errors or [])
            raise CommandError(
                f"Import #{import_row.id} failed with {error_count} recorded error(s)."
            )

        self.stdout.write(self.style.SUCCESS(
            f"Import #{import_row.id} completed: stats={import_row.stats}"
        ))
        if import_row.errors:
            self.stdout.write(self.style.WARNING(
                f"  {len(import_row.errors)} warning(s) recorded."
            ))


def _sha256_of_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
