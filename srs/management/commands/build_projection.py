"""Build a two-dimensional embedding of the reference library."""

import json
from math import log10

from django.core.management.base import BaseCommand, CommandError

from ...benchmarks import load_signatures
from ...projections import fit_pca, fit_tsne


class Command(BaseCommand):
    help = (
        "Embed the reference library in two dimensions for visualisation. "
        "PCA is fast and is what the live analysis uses; t-SNE separates "
        "clusters better but takes minutes and cannot place a new sample."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--method",
            choices=("pca", "tsne"),
            default="pca",
        )
        parser.add_argument(
            "--output",
            required=True,
            help="Path to write the embedding to, as JSON.",
        )
        parser.add_argument(
            "--max-samples",
            type=int,
            default=0,
            help="Cap the number of samples embedded. 0 uses all of them.",
        )
        parser.add_argument("--perplexity", type=float, default=30.0)
        parser.add_argument("--iterations", type=int, default=500)

    def handle(self, *args, **options):
        method = options["method"]

        self.stdout.write("Loading reference library...")
        signatures = load_signatures({"handle_missing": "half_dl"})
        if not signatures:
            raise CommandError("No reference samples found. Import the library first.")

        limit = options["max_samples"]
        if limit and limit < len(signatures):
            # Even spacing gives the same subset on every run.
            step = len(signatures) / limit
            signatures = [signatures[int(index * step)] for index in range(limit)]

        self.stdout.write(f"  embedding {len(signatures)} samples with {method}")

        if method == "pca":
            points = self.build_pca(signatures)
        else:
            points = self.build_tsne(signatures, options)

        payload = {
            "method": method,
            "n_points": len(points),
            "points": points,
        }
        with open(options["output"], "w", encoding="utf-8") as handle:
            json.dump(payload, handle)

        self.stdout.write(self.style.SUCCESS(
            f"Wrote {len(points)} points to {options['output']}"
        ))

    def build_pca(self, signatures):
        model = fit_pca([signature["values"] for signature in signatures])
        if model is None:
            raise CommandError("Too few shared elements to fit a projection.")

        self.stdout.write(
            "  explained variance: "
            + ", ".join(f"{ratio:.1%}" for ratio in model.explained_variance_ratio)
        )

        points = []
        for signature in signatures:
            position = model.project(signature["values"])
            if position is None:
                continue
            points.append({
                "id": signature["id"],
                "x": position[0],
                "y": position[1],
                "kind": "reference",
                "deposit_class": signature["deposit_class"],
            })
        return points

    def build_tsne(self, signatures, options):
        # t-SNE needs a dense matrix, so the same CLR construction the live
        # projection uses is applied first.
        model = fit_pca([signature["values"] for signature in signatures])
        if model is None:
            raise CommandError("Too few shared elements to build an embedding.")

        symbols = model.symbols
        vectors = []
        kept = []
        for signature in signatures:
            values = signature["values"]
            present = [values.get(symbol) for symbol in symbols]
            measured = [v for v in present if v is not None and v > 0]
            if len(measured) < max(2, len(symbols) // 2):
                continue

            centre = sum(log10(v) for v in measured) / len(measured)
            vectors.append([
                (log10(v) - centre) if (v is not None and v > 0) else mean
                for v, mean in zip(present, model.means)
            ])
            kept.append(signature)

        def progress(done, total):
            self.stdout.write(f"    iteration {done}/{total}")

        embedding = fit_tsne(
            vectors,
            perplexity=options["perplexity"],
            iterations=options["iterations"],
            progress=progress,
        )

        return [
            {
                "id": signature["id"],
                "x": position[0],
                "y": position[1],
                "kind": "reference",
                "deposit_class": signature["deposit_class"],
            }
            for signature, position in zip(kept, embedding)
        ]
