"""Upload, search and analysis endpoints for the SRS API."""

import hashlib
import heapq
import json
import logging
import threading
from math import ceil
from time import perf_counter

from django.db import close_old_connections, transaction
from django.db.models import Count
from django.utils import timezone
from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import OpenApiParameter, extend_schema
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from .algorithms import available_algorithms, default_algorithm_id, get_algorithm
from .algorithms.base import PairwiseSimilarity
from .algorithms.knn_aitchison import DEFAULT_DETAIL_TOP_N, DEFAULT_K
from .authentication import caller_audit_fields
from .importers import run_import, run_dataset_import
from .preprocessing import (
    PIPELINE_VERSION,
    extract_values,
    normalise_symbol,
    prepare_vectors,
    resolve_options,
)
from .models import (
    Dataset,
    FullAnalysis,
    FullAnalysisMatch,
    ReferenceImport,
    ReferenceSample,
    Sample,
    SampleMeasurement,
)
from .projections import fit_pca, project_points
from .serializers import (
    DatasetSerializer,
    DatasetUploadSerializer,
    ReferenceImportSerializer,
    ReferenceImportUploadSerializer,
    ReferenceLibrarySearchResultSerializer,
)


logger = logging.getLogger(__name__)


def _as_boolean(value) -> bool:
    """Convert common request values to a real boolean."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _as_json_object(value) -> dict:
    """Return a JSON object from current or older stored values."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def reference_library_version() -> str:
    """Identify the latest completed reference-library import."""
    latest = (
        ReferenceImport.objects
        .filter(status=ReferenceImport.STATUS_COMPLETED)
        .order_by("-completed_at", "-id")
        .values_list("id", "source_name")
        .first()
    )

    if not latest:
        return ""

    import_id, source_name = latest
    return f"{import_id}:{source_name}"


def _sha256_of(uploaded_file) -> str:
    """Hash an upload, then rewind it before Django saves the file."""
    hasher = hashlib.sha256()
    for chunk in uploaded_file.chunks():
        hasher.update(chunk)
    uploaded_file.seek(0)
    return hasher.hexdigest()


class ReferenceImportViewSet(viewsets.ModelViewSet):
    """Upload, inspect, rerun or remove a reference-library import."""

    queryset = ReferenceImport.objects.all().order_by("-created_at")
    parser_classes = (MultiPartParser, FormParser)
    http_method_names = ["get", "post", "delete", "head", "options"]

    def get_serializer_class(self):
        if self.action == "create":
            return ReferenceImportUploadSerializer
        return ReferenceImportSerializer

    def create(self, request, *args, **kwargs):
        upload = self.get_serializer(data=request.data)
        upload.is_valid(raise_exception=True)

        data_file = upload.validated_data["data_file"]
        metadata_file = upload.validated_data["metadata_file"]
        data_hash = _sha256_of(data_file)
        metadata_hash = _sha256_of(metadata_file)

        existing = ReferenceImport.objects.filter(
            data_sha256=data_hash,
            metadata_sha256=metadata_hash,
        ).exclude(status=ReferenceImport.STATUS_FAILED).first()
        if existing is not None:
            return Response(
                ReferenceImportSerializer(existing).data,
                status=status.HTTP_200_OK,
            )

        with transaction.atomic():
            import_row = upload.save(
                data_sha256=data_hash,
                metadata_sha256=metadata_hash,
                status=ReferenceImport.STATUS_PENDING,
                **caller_audit_fields(
                    request,
                    "uploaded_by_id",
                    "uploaded_by_email",
                ),
            )

        threading.Thread(target=run_import, args=[import_row.id], daemon=True).start()

        return Response(
            ReferenceImportSerializer(import_row).data,
            status=status.HTTP_202_ACCEPTED,
        )

    @extend_schema(request=None, responses={202: ReferenceImportSerializer})
    @action(detail=True, methods=["post"], url_path="rerun")
    def rerun(self, request, pk=None):
        """Re-parse the stored workbook pair without re-uploading."""
        import_row = self.get_object()
        import_row.status = ReferenceImport.STATUS_PENDING
        import_row.errors = []
        import_row.stats = {}
        import_row.completed_at = None
        import_row.save(update_fields=["status", "errors", "stats", "completed_at"])

        threading.Thread(target=run_import, args=[import_row.id], daemon=True).start()

        return Response(
            ReferenceImportSerializer(import_row).data,
            status=status.HTTP_202_ACCEPTED,
        )

def _delete_failed_dataset(dataset):
    """Remove a failed initial dataset upload from the database and file storage."""

    stored_file = dataset.uploaded_file

    with transaction.atomic():
        # Explicitly remove any partially imported data.
        SampleMeasurement.objects.filter(
            sample__dataset=dataset
        ).delete()

        Sample.objects.filter(
            dataset=dataset
        ).delete()

        dataset.delete()

    # Also remove the uploaded CSV from file storage.
    if stored_file:
        try:
            stored_file.delete(save=False)
        except Exception:
            logger.exception(
                "Dataset records were removed, but uploaded file cleanup failed."
            )


class DatasetViewSet(viewsets.ModelViewSet):
    """Upload, inspect, rerun or remove an analysed CSV dataset."""

    queryset = Dataset.objects.all().order_by("-created_at")
    parser_classes = (MultiPartParser, FormParser, JSONParser)

    def get_serializer_class(self):
        if self.action == "create":
            return DatasetUploadSerializer
        return DatasetSerializer

    def create(self, request, *args, **kwargs):
        upload = self.get_serializer(data=request.data)
        upload.is_valid(raise_exception=True)

        uploaded_file = upload.validated_data["uploaded_file"]
        file_hash = _sha256_of(uploaded_file)

        existing = (
            Dataset.objects
            .filter(file_sha256=file_hash)
            .exclude(status=Dataset.STATUS_FAILED)
            .first()
        )

        if existing is not None:
            return Response(
                DatasetSerializer(existing).data,
                status=status.HTTP_200_OK,
            )

        with transaction.atomic():
            dataset = upload.save(
                original_filename=uploaded_file.name,
                file_sha256=file_hash,
                status=Dataset.STATUS_PENDING,
                **caller_audit_fields(
                    request,
                    "uploaded_by_id",
                    "uploaded_by_email",
                ),
            )

        try:
            run_dataset_import(dataset.id)

        except ValueError as exc:
            # Invalid CSV/data supplied by the user.
            logger.warning(
                "Dataset validation failed for dataset %s: %s",
                dataset.id,
                exc,
            )

            _delete_failed_dataset(dataset)

            return Response(
                {
                    "error": "Dataset validation failed.",
                    "details": str(exc),
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        except Exception:
            # Unexpected server/database/programming error.
            logger.exception(
                "Unexpected dataset import failure for dataset %s",
                dataset.id,
            )

            _delete_failed_dataset(dataset)

            return Response(
                {
                    "error": (
                        "The dataset could not be uploaded because "
                        "an unexpected server error occurred."
                    )
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        dataset.refresh_from_db()

        return Response(
            DatasetSerializer(dataset).data,
            status=status.HTTP_201_CREATED,
        )

    @action(detail=True, methods=["post"], url_path="rerun")
    def rerun(self, request, pk=None):
        dataset = self.get_object()

        try:
            run_dataset_import(dataset.id)

        except Exception:
            logger.exception(
                "Dataset rerun failed for dataset %s",
                dataset.id,
            )

            dataset.refresh_from_db()

            return Response(
                DatasetSerializer(dataset).data,
                status=status.HTTP_400_BAD_REQUEST,
            )

        dataset.refresh_from_db()

        return Response(
            DatasetSerializer(dataset).data
        )

    @action(detail=True, methods=["get"], url_path="data")
    def data(self, request, pk=None):
        dataset = self.get_object()

        try:
            page = int(request.query_params.get("page", 1))
            page_size = int(request.query_params.get("page_size", 50))

        except ValueError:
            return Response(
                {
                    "error": "page and page_size must be integers."
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        page = max(page, 1)
        page_size = max(min(page_size, 500), 1)

        samples_qs = (
            Sample.objects
            .filter(dataset=dataset)
            .prefetch_related("measurements__element")
            .order_by("id")
        )

        total_rows = samples_qs.count()
        total_pages = max(
            1,
            ceil(total_rows / page_size),
        )

        start = (page - 1) * page_size
        end = start + page_size

        page_samples = list(
            samples_qs[start:end]
        )

        measurement_fields = (
            SampleMeasurement.objects
            .filter(sample__dataset=dataset)
            .values_list(
                "element__symbol",
                "unit",
            )
            .order_by(
                "element__symbol",
                "unit",
            )
            .distinct()
        )

        measurement_columns = [
            f"{symbol}_{unit}" if unit else symbol
            for symbol, unit in measurement_fields
        ]

        columns = [
            "sample_id",
            "latitude",
            "longitude",
        ] + measurement_columns

        rows = []

        for sample in page_samples:
            row = {
                "sample_id": sample.sample_code,
                "latitude": sample.latitude,
                "longitude": sample.longitude,
                "_sample_db_id": sample.id,
                "_measurement_ids": {},
            }

            for column in measurement_columns:
                row[column] = None

            for measurement in sample.measurements.all():
                column_name = self._measurement_column_name(
                    measurement
                )

                row[column_name] = measurement.value

                row["_measurement_ids"][
                    column_name
                ] = measurement.id

            rows.append(row)

        null_counts = {}

        for column in columns:
            null_counts[column] = sum(
                1
                for row in rows
                if (
                    row.get(column) is None
                    or row.get(column) == ""
                )
            )

        return Response(
            {
                "dataset_id": dataset.id,
                "dataset_name": dataset.name,
                "columns": columns,
                "rows": rows,
                "total_rows": total_rows,
                "total_pages": total_pages,
                "page": page,
                "page_size": page_size,
                "null_counts": null_counts,
            }
        )

    def _measurement_column_name(self, measurement):
        symbol = measurement.element.symbol

        if measurement.unit:
            return f"{symbol}_{measurement.unit}"

        return symbol

# Reference-library search filters shown in the generated API documentation.
@extend_schema(
    parameters=[
        OpenApiParameter(
            "deposit_name",
            OpenApiTypes.STR,
            description="Partial deposit name, for example Olympic.",
        ),
        OpenApiParameter(
            "deposit_type",
            OpenApiTypes.STR,
            description="Exact deposit type, for example IOCG or Epithermal.",
        ),
        OpenApiParameter(
            "mineral_system",
            OpenApiTypes.STR,
            description="Exact mineral system, for example Orogenic Au.",
        ),
        OpenApiParameter(
            "country",
            OpenApiTypes.STR,
            description="Partial country name, for example Australia.",
        ),
        OpenApiParameter(
            "state_region",
            OpenApiTypes.STR,
            description="Partial state or region, for example Queensland.",
        ),
        OpenApiParameter(
            "sample_code",
            OpenApiTypes.STR,
            description="Partial sample code, for example 700001.",
        ),
        OpenApiParameter(
            "sample_type",
            OpenApiTypes.STR,
            description="Exact sample type, for example VHMS or Skarn Au.",
        ),
        OpenApiParameter(
            "ore_minerals",
            OpenApiTypes.STR,
            description="Comma-separated minerals that must all be present.",
        ),
        OpenApiParameter(
            "element",
            OpenApiTypes.STR,
            description="Element symbol used for measurement filters.",
        ),
        OpenApiParameter(
            "min_value",
            OpenApiTypes.FLOAT,
            description="Minimum value for the selected element.",
        ),
        OpenApiParameter(
            "max_value",
            OpenApiTypes.FLOAT,
            description="Maximum value for the selected element.",
        ),
        OpenApiParameter(
            "analytical_method",
            OpenApiTypes.STR,
            description="Exact analytical method, for example FA or AR.",
        ),
        OpenApiParameter(
            "exclude_bdl",
            OpenApiTypes.BOOL,
            description="Exclude below-detection-limit measurements.",
        ),
        OpenApiParameter(
            "limit",
            OpenApiTypes.INT,
            description="Results to return (default 25, maximum 500).",
        ),
        OpenApiParameter(
            "offset",
            OpenApiTypes.INT,
            description="Results to skip for pagination.",
        ),
    ],
    responses={200: OpenApiTypes.OBJECT},
)
class ReferenceLibrarySearchView(APIView):
    """Search reference samples and their measurements."""

    def get(self, request):
        qs = (
            ReferenceSample.objects
            .select_related("reference_deposit")
            .prefetch_related("measurements__element")
        )

        p = request.query_params

        # Deposit-level filters
        if deposit_name := p.get("deposit_name"):
            qs = qs.filter(reference_deposit__name__icontains=deposit_name)
        if deposit_type := p.get("deposit_type"):
            qs = qs.filter(reference_deposit__deposit_type__iexact=deposit_type)
        if mineral_system := p.get("mineral_system"):
            qs = qs.filter(reference_deposit__mineral_system__iexact=mineral_system)
        if country := p.get("country"):
            qs = qs.filter(reference_deposit__country__icontains=country)
        if state_region := p.get("state_region"):
            qs = qs.filter(reference_deposit__state_region__icontains=state_region)

        # Sample-level filters
        if sample_code := p.get("sample_code"):
            qs = qs.filter(sample_code__icontains=sample_code)
        if sample_type := p.get("sample_type"):
            qs = qs.filter(sample_type__iexact=sample_type)
        if ore_minerals := p.get("ore_minerals"):
            for mineral in ore_minerals.split(","):
                # Text search also works with the local SQLite database.
                qs = qs.filter(metadata__icontains=f'"{mineral.strip()}"')

        # Measurement-level filters
        if element := p.get("element"):
            filters = {"measurements__element__symbol__iexact": element}
            try:
                if min_value := p.get("min_value"):
                    filters["measurements__value__gte"] = float(min_value)
                if max_value := p.get("max_value"):
                    filters["measurements__value__lte"] = float(max_value)
            except (TypeError, ValueError):
                return Response(
                    {"error": "min_value and max_value must be numbers."},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            if method := p.get("analytical_method"):
                filters["measurements__analytical_method__iexact"] = method
            if _as_boolean(p.get("exclude_bdl", False)):
                filters["measurements__below_detection_limit"] = False
            qs = qs.filter(**filters)

        qs = qs.distinct()

        try:
            limit = max(1, min(int(p.get("limit", 25)), 500))
            offset = max(0, int(p.get("offset", 0)))
        except (TypeError, ValueError):
            return Response(
                {"error": "limit and offset must be integers."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        total = qs.count()
        samples = qs[offset : offset + limit]

        return Response(
            {
                "total": total,
                "limit": limit,
                "offset": offset,
                "results": ReferenceLibrarySearchResultSerializer(
                    samples,
                    many=True,
                ).data,
            }
        )


class BulkReferenceSampleDetailView(APIView):
    """Return details for up to 50 reference samples in one request."""

    @extend_schema(
        request=OpenApiTypes.OBJECT,
        responses={200: OpenApiTypes.OBJECT},
    )
    def post(self, request):
        requested_ids = request.data.get("ids")

        if not isinstance(requested_ids, list) or not requested_ids:
            return Response(
                {"error": "ids must be a non-empty list."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            # dict.fromkeys removes duplicates without changing requested order.
            sample_ids = list(
                dict.fromkeys(int(sample_id) for sample_id in requested_ids)
            )
        except (TypeError, ValueError):
            return Response(
                {"error": "Every sample ID must be an integer."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if len(sample_ids) > 50:
            return Response(
                {"error": "A maximum of 50 sample IDs can be requested."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        samples = (
            ReferenceSample.objects
            .filter(id__in=sample_ids)
            .select_related("reference_deposit")
            .prefetch_related("measurements__element")
        )
        samples_by_id = {sample.id: sample for sample in samples}

        # Preserve the ranked ID order supplied by the results page.
        results = [
            self.serialize_sample(samples_by_id[sample_id])
            for sample_id in sample_ids
            if sample_id in samples_by_id
        ]

        return Response({"results": results})

    def serialize_sample(self, sample):
        deposit = sample.reference_deposit

        return {
            "id": sample.id,
            "sample_code": sample.sample_code,
            "sample_type": sample.sample_type,
            "latitude": (
                sample.latitude
                if sample.latitude is not None
                else (deposit.latitude if deposit else None)
            ),
            "longitude": (
                sample.longitude
                if sample.longitude is not None
                else (deposit.longitude if deposit else None)
            ),
            "source_dataset": sample.source_dataset,
            "source_reference": sample.source_reference,
            "metadata": sample.metadata,
            "deposit_name": deposit.name if deposit else None,
            "deposit_type": deposit.deposit_type if deposit else None,
            "measurements": [
                {
                    "element_symbol": measurement.element.symbol,
                    "value": measurement.value,
                    "unit": measurement.unit,
                    "analytical_method": measurement.analytical_method,
                    "below_detection_limit": measurement.below_detection_limit,
                    "detection_limit": measurement.detection_limit,
                }
                for measurement in sample.measurements.all()
            ],
        }


class SimilarityAlgorithmListView(APIView):
    """List the similarity algorithms available to the analysis page."""

    @extend_schema(
        responses={200: OpenApiTypes.OBJECT},
        description=(
            "Similarity algorithms available to this deployment. Each result "
            "carries an id usable as similarity_method when creating a full "
            "analysis, its version, the optional result blocks it can produce, "
            "and whether it is the configured default."
        ),
    )
    def get(self, request):
        algorithms = available_algorithms()

        return Response({
            "count": len(algorithms),
            "default": next(
                (
                    algorithm["id"]
                    for algorithm in algorithms
                    if algorithm["is_default"]
                ),
                None,
            ),
            "results": algorithms,
        })


class FullAnalysisListCreateView(APIView):
    """List saved analyses or queue a new multi-sample analysis."""

    def normalise_element_symbol(self, symbol):
        """Return a consistently capitalised element symbol."""
        return normalise_symbol(symbol)

    def serialize_full_analysis_summary(self, full_analysis):
        sample_data = _as_json_object(full_analysis.sample_data)
        parameters = _as_json_object(full_analysis.parameters)
        samples = sample_data.get("samples") or []
        match_count = getattr(full_analysis, "saved_match_count", None)
        if match_count is None:
            match_count = full_analysis.ranked_matches.count()

        return {
            "id": full_analysis.id,
            "name": full_analysis.name,
            "uploaded_sample_code": full_analysis.uploaded_sample_code,
            "source_filename": full_analysis.source_filename,
            "method": full_analysis.method,
            "status": full_analysis.status,
            "created_at": full_analysis.created_at,
            "completed_at": full_analysis.completed_at,
            "match_count": match_count,
            "sample_count": len(samples) or 1,
            "sample_codes": [
                sample.get("sample_code")
                for sample in samples
                if isinstance(sample, dict) and sample.get("sample_code")
            ] or [full_analysis.uploaded_sample_code],
            "selected_elements": parameters.get("selected_elements") or [],
            "preprocessing": parameters.get("preprocessing") or {},
            "parameters": parameters,
        }

    @extend_schema(
        operation_id="full_analysis_list",
        responses={200: OpenApiTypes.OBJECT},
    )
    def get(self, request):
        full_analyses = (
            FullAnalysis.objects
            .annotate(saved_match_count=Count("ranked_matches"))
            .order_by("-created_at")
        )

        return Response({
            "count": full_analyses.count(),
            "results": [
                self.serialize_full_analysis_summary(full_analysis)
                for full_analysis in full_analyses
            ],
        })

    @extend_schema(
        operation_id="full_analysis_create",
        request=OpenApiTypes.OBJECT,
        responses={202: OpenApiTypes.OBJECT},
    )
    def post(self, request):
        analysis_name = (
            request.data.get("analysis_name")
            or request.data.get("name")
            or "Uploaded analysis"
        )
        source_filename = request.data.get("source_filename", "")
        similarity_method = (
            request.data.get("similarity_method")
            or default_algorithm_id()
        )
        preprocessing = request.data.get("preprocessing") or {}

        requested_top_n = request.data.get("top_n", 200)
        if str(requested_top_n).strip().lower() == "all":
            # Persist the complete ranking while result retrieval remains
            # paginated. Resolve this to a number now so the background worker
            # sees a stable limit for the reference-library snapshot it runs.
            top_n = ReferenceSample.objects.count()
        else:
            try:
                top_n = int(requested_top_n)
            except (TypeError, ValueError):
                return Response(
                    {"error": "top_n must be an integer or 'all'."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

        try:
            detail_top_n = max(
                0,
                int(request.data.get("detail_top_n", DEFAULT_DETAIL_TOP_N)),
            )
            neighbours = max(1, int(request.data.get("k", DEFAULT_K)))
        except (TypeError, ValueError):
            return Response(
                {"error": "detail_top_n and k must be integers."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        raw_samples = request.data.get("samples")

        # Continue accepting the original single-sample request shape.
        if not isinstance(raw_samples, list):
            raw_samples = [{
                "sample_code": request.data.get("sample_code"),
                "name": request.data.get("sample_name") or request.data.get("name"),
                "latitude": request.data.get("latitude"),
                "longitude": request.data.get("longitude"),
                "measurements": request.data.get("measurements"),
            }]

        samples = [
            self.normalise_analysed_sample(sample, index)
            for index, sample in enumerate(raw_samples)
            if isinstance(sample, dict)
        ]

        if not samples:
            return Response(
                {"error": "You must provide at least one sample."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        invalid_samples = [
            sample["sample_code"]
            for sample in samples
            if not sample["measurements"]
        ]
        if invalid_samples:
            return Response(
                {
                    "error": "Every sample must contain at least one valid measurement.",
                    "samples": invalid_samples,
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Keep the value positive, but do not impose an upper limit. Large result
        # sets are exposed through the paginated per-sample results endpoint.
        top_n = max(1, top_n)
        first_sample = samples[0]
        sample_snapshot = dict(request.data)
        sample_snapshot["samples"] = samples

        # Resolve once so every saved field names the algorithm that will run.
        requested_similarity_method = similarity_method
        resolved_algorithm = get_algorithm(similarity_method)
        similarity_method = resolved_algorithm.id

        full_analysis = FullAnalysis.objects.create(
            name=analysis_name,
            uploaded_sample_code=first_sample["sample_code"],
            source_filename=source_filename,
            sample_data=sample_snapshot,
            method=similarity_method,
            parameters={
                "top_n": top_n,
                "batch_size": 250,
                "detail_top_n": detail_top_n,
                "k": neighbours,
                "dataset_id": request.data.get("dataset_id"),
                "dataset_name": request.data.get("dataset_name"),
                "selected_elements": request.data.get("selected_elements") or [],
                "preprocessing": preprocessing,
                "similarity_method": similarity_method,
                "requested_similarity_method": requested_similarity_method,
                "algorithm_id": resolved_algorithm.id,
                "algorithm_version": resolved_algorithm.version,
                "pipeline_version": PIPELINE_VERSION,
                "samples_completed": 0,
                "sample_count": len(samples),
                "references_processed": 0,
                "reference_count": ReferenceSample.objects.count(),
                "note": "Background batched similarity analysis.",
            },
            status=FullAnalysis.STATUS_PENDING,
            **caller_audit_fields(
                request,
                "created_by_id",
                "created_by_email",
            ),
        )

        # Return immediately instead of holding an HTTP request open for the
        # complete calculation. Each thread opens its own database connection.
        threading.Thread(
            target=self.process_full_analysis,
            args=(full_analysis.id,),
            daemon=True,
        ).start()

        return Response(
            {
                "message": "Full analysis queued.",
                "full_analysis_id": full_analysis.id,
                "analysed_sample_count": len(samples),
                "status": FullAnalysis.STATUS_PENDING,
                "results_url": f"/api/full-analysis/{full_analysis.id}/",
            },
            status=status.HTTP_202_ACCEPTED,
        )

    def process_full_analysis(self, full_analysis_id):
        """Process every tested sample in bounded reference-library batches."""
        close_old_connections()

        try:
            full_analysis = FullAnalysis.objects.get(id=full_analysis_id)
            sample_data = _as_json_object(full_analysis.sample_data)
            samples = sample_data.get("samples") or []
            parameters = _as_json_object(full_analysis.parameters)
            top_n = int(parameters["top_n"])
            batch_size = int(parameters.get("batch_size", 250))
            similarity_method = parameters.get(
                "similarity_method",
                default_algorithm_id(),
            )
            preprocessing = parameters.get("preprocessing") or {}
            algorithm = get_algorithm(similarity_method)

            # Provenance is written before the work starts, so a run that fails
            # partway still records what was scoring it.
            full_analysis.status = FullAnalysis.STATUS_RUNNING
            full_analysis.algorithm_id = algorithm.id
            full_analysis.algorithm_version = algorithm.version
            full_analysis.pipeline_version = PIPELINE_VERSION
            full_analysis.reference_library_version = reference_library_version()
            full_analysis.save(update_fields=[
                "status",
                "algorithm_id",
                "algorithm_version",
                "pipeline_version",
                "reference_library_version",
            ])
            started = perf_counter()

            for sample_index, sample in enumerate(samples):
                parameters.update({
                    "current_sample_index": sample_index,
                    "references_processed": 0,
                })
                full_analysis.parameters = parameters
                full_analysis.save(update_fields=["parameters"])

                self.create_ranked_matches(
                    full_analysis,
                    sample["measurements"],
                    sample_index,
                    top_n,
                    batch_size,
                    similarity_method,
                    preprocessing,
                )
                parameters["samples_completed"] = sample_index + 1
                parameters["references_processed"] = parameters["reference_count"]
                full_analysis.parameters = parameters
                full_analysis.save(update_fields=["parameters"])

            full_analysis.status = FullAnalysis.STATUS_COMPLETED
            full_analysis.completed_at = timezone.now()
            full_analysis.runtime_ms = (perf_counter() - started) * 1000
            full_analysis.sample_results = self.build_sample_results(
                full_analysis,
                samples,
            )
            full_analysis.projection = self.build_projection(
                full_analysis,
                samples,
                resolve_options(
                    preprocessing,
                    parameters.get("selected_elements"),
                ),
            )
            full_analysis.save(update_fields=[
                "status",
                "completed_at",
                "runtime_ms",
                "sample_results",
                "projection",
            ])
        except Exception as error:
            logger.exception("Full analysis %s failed", full_analysis_id)
            saved_parameters = (
                FullAnalysis.objects
                .filter(id=full_analysis_id)
                .values_list("parameters", flat=True)
                .first()
            )
            FullAnalysis.objects.filter(id=full_analysis_id).update(
                status=FullAnalysis.STATUS_FAILED,
                completed_at=timezone.now(),
                parameters={
                    **_as_json_object(saved_parameters),
                    "error": str(error),
                },
            )
        finally:
            close_old_connections()

    def build_sample_results(self, full_analysis, samples):
        """Summarise each sample and its five leading matches."""
        leading = (
            full_analysis.ranked_matches
            .filter(rank__lte=5)
            .select_related("reference_sample__reference_deposit")
            .order_by("analysed_sample_index", "rank")
        )
        by_sample = {}

        for match in leading:
            reference_sample = match.reference_sample
            deposit = (
                reference_sample.reference_deposit
                if reference_sample is not None
                else None
            )
            by_sample.setdefault(match.analysed_sample_index, []).append({
                "reference_sample_id": match.reference_sample_id,
                "deposit_id": (deposit.three_char_code if deposit else ""),
                "deposit_name": (deposit.name if deposit else ""),
                "similarity": match.similarity_score,
            })

        return [
            {
                "sample_id": sample.get("sample_code"),
                "lat": sample.get("latitude"),
                "lon": sample.get("longitude"),
                "top_matches": by_sample.get(sample_index, []),
            }
            for sample_index, sample in enumerate(samples)
        ]

    def build_projection(self, full_analysis, samples, options, max_references=200):
        """Build a small PCA plot from analysed samples and leading matches."""
        matched_ids = list(
            full_analysis.ranked_matches
            .filter(rank__lte=max_references)
            .values_list("reference_sample_id", flat=True)
            .distinct()
        )
        if not matched_ids:
            return None

        reference_values = {}
        for reference_sample in (
            ReferenceSample.objects
            .filter(id__in=matched_ids)
            .prefetch_related("measurements__element")
        ):
            values, _ = extract_values(reference_sample.measurements.all(), options)
            if values:
                reference_values[reference_sample.id] = values

        sample_values = {}
        for sample_index, sample in enumerate(samples):
            values, _ = extract_values(sample.get("measurements") or [], options)
            if values:
                sample_values[sample_index] = values

        model = fit_pca(list(reference_values.values()) + list(sample_values.values()))
        if model is None:
            return None

        points = project_points(
            model,
            [
                (str(sample.get("sample_code") or index), "sample", values)
                for index, (sample, values) in enumerate(
                    (samples[i], v) for i, v in sample_values.items()
                )
            ] + [
                (str(reference_id), "reference", values)
                for reference_id, values in reference_values.items()
            ],
        )
        if not points:
            return None

        return {
            "method": "pca",
            "elements_used": model.symbols,
            "explained_variance_ratio": model.explained_variance_ratio,
            "points": points,
        }

    def normalise_analysed_sample(self, sample, sample_index):
        """Return one stable JSON representation of a tested CSV row."""
        sample_code = (
            sample.get("sample_code")
            or sample.get("sample_id")
            or sample.get("name")
            or f"Sample {sample_index + 1}"
        )
        measurements = []

        for item in sample.get("measurements") or []:
            if not isinstance(item, dict):
                continue

            symbol = self.normalise_element_symbol(
                item.get("element_symbol") or item.get("symbol")
            )
            if not symbol:
                continue

            measurements.append({
                "element_symbol": symbol,
                "value": item.get("value"),
                "unit": item.get("unit", "ppm"),
                "below_detection_limit": _as_boolean(
                    item.get("below_detection_limit", False)
                ),
                "detection_limit": item.get("detection_limit"),
            })

        return {
            **sample,
            "sample_code": str(sample_code),
            "name": sample.get("name") or str(sample_code),
            "latitude": sample.get("latitude"),
            "longitude": sample.get("longitude"),
            "measurements": measurements,
        }

    def create_ranked_matches(
        self,
        full_analysis,
        measurements,
        sample_index,
        top_n,
        batch_size,
        similarity_method,
        preprocessing,
    ):
        """Compare one analysed sample with the reference library."""
        # Resolved once, outside the reference loop, so policy names are not
        # revalidated on every one of a thousand comparisons.
        saved_parameters = _as_json_object(full_analysis.parameters)
        options = resolve_options(
            preprocessing,
            saved_parameters.get("selected_elements"),
        )
        algorithm = get_algorithm(similarity_method)

        # An algorithm that needs the whole library at once cannot be driven by
        # the streaming heap below, so it gets its own path. Without this a
        # non-pairwise algorithm would be selectable and then silently fail.
        if not isinstance(algorithm, PairwiseSimilarity):
            return self.create_ranked_matches_via_compare(
                algorithm,
                full_analysis,
                measurements,
                sample_index,
                top_n,
                batch_size,
                options,
            )

        # A symbol-to-value dictionary makes finding shared elements inexpensive.
        # The pipeline handles unit conversion, the censored-data policy, and
        # the user's element selection.
        input_values, input_imputed = extract_values(measurements, options)

        # Keep one global bounded heap across every batch. This guarantees the
        # final ranking is the best top_n from the complete reference library,
        # rather than a separate partial ranking from each batch.
        best_matches = []
        reference_count = ReferenceSample.objects.count()
        FullAnalysisMatch.objects.filter(
            full_analysis=full_analysis,
            analysed_sample_index=sample_index,
        ).delete()

        for offset in range(0, reference_count, batch_size):
            reference_batch = list(
                ReferenceSample.objects
                .select_related("reference_deposit")
                .prefetch_related("measurements__element")
                .order_by("id")[offset:offset + batch_size]
            )

            for reference_sample in reference_batch:
                reference_values, reference_imputed = extract_values(
                    reference_sample.measurements.all(),
                    options,
                )

                common_elements = set(input_values) & set(reference_values)
                if not common_elements:
                    continue

                score = self.calculate_similarity_score(
                    input_values,
                    reference_values,
                    common_elements,
                    similarity_method,
                    options,
                    input_imputed | reference_imputed,
                )
                candidate = (score, reference_sample.id)

                if len(best_matches) < top_n:
                    heapq.heappush(best_matches, candidate)
                elif candidate > best_matches[0]:
                    heapq.heapreplace(best_matches, candidate)

            parameters = _as_json_object(
                FullAnalysis.objects
                .filter(id=full_analysis.id)
                .values_list("parameters", flat=True)
                .first()
            )
            parameters.update({
                "current_sample_index": sample_index,
                "references_processed": min(offset + batch_size, reference_count),
                "reference_count": reference_count,
            })
            FullAnalysis.objects.filter(id=full_analysis.id).update(
                parameters=parameters,
            )

        ranked_candidates = sorted(best_matches, reverse=True)
        saved_parameters = _as_json_object(full_analysis.parameters)
        detail_top_n = int(
            saved_parameters.get("detail_top_n", DEFAULT_DETAIL_TOP_N)
        )
        neighbours = int(saved_parameters.get("k", DEFAULT_K))
        detail = self.build_match_detail(
            algorithm,
            input_values,
            input_imputed,
            ranked_candidates[:detail_top_n],
            options,
            nearest_candidates=ranked_candidates[:neighbours],
        )

        created_matches = [
            FullAnalysisMatch(
                full_analysis=full_analysis,
                reference_sample_id=reference_sample_id,
                analysed_sample_index=sample_index,
                rank=rank,
                similarity_score=score,
                **detail.get(reference_sample_id, {}),
            )
            for rank, (score, reference_sample_id)
            in enumerate(ranked_candidates, start=1)
        ]
        FullAnalysisMatch.objects.bulk_create(
            created_matches,
            batch_size=batch_size,
        )

        return created_matches

    def create_ranked_matches_via_compare(
        self,
        algorithm,
        full_analysis,
        measurements,
        sample_index,
        top_n,
        batch_size,
        options,
    ):
        """Run an algorithm that compares against the whole library at once."""
        input_values, input_imputed = extract_values(measurements, options)

        FullAnalysisMatch.objects.filter(
            full_analysis=full_analysis,
            analysed_sample_index=sample_index,
        ).delete()

        references = []
        for reference_sample in (
            ReferenceSample.objects
            .select_related("reference_deposit")
            .prefetch_related("measurements__element")
            .order_by("id")
            .iterator(chunk_size=batch_size)
        ):
            values, imputed = extract_values(
                reference_sample.measurements.all(),
                options,
            )
            if not values:
                continue

            deposit = reference_sample.reference_deposit
            references.append({
                "id": reference_sample.id,
                "values": values,
                "imputed": imputed,
                "deposit_id": (deposit.three_char_code or deposit.name) if deposit else "",
                "deposit_name": deposit.name if deposit else "",
                "deposit_class": (deposit.deposit_type or "") if deposit else "",
            })

        sample_data = _as_json_object(full_analysis.sample_data)
        saved_samples = sample_data.get("samples") or []
        analysed_sample = (
            saved_samples[sample_index]
            if sample_index < len(saved_samples)
            else {}
        )
        sample_code = (
            analysed_sample.get("sample_code")
            or f"Sample {sample_index + 1}"
        )

        saved_parameters = _as_json_object(full_analysis.parameters)
        result = algorithm.compare(
            [{
                "sample_code": sample_code,
                "values": input_values,
                "imputed": input_imputed,
            }],
            references,
            {
                "top_n": top_n,
                "preprocessing": options,
                "detail_top_n": saved_parameters.get(
                    "detail_top_n",
                    DEFAULT_DETAIL_TOP_N,
                ),
                "k": saved_parameters.get("k", DEFAULT_K),
            },
        )

        created_matches = [
            FullAnalysisMatch(
                full_analysis=full_analysis,
                reference_sample_id=match.reference_sample_id,
                analysed_sample_index=sample_index,
                rank=match.rank,
                similarity_score=float(match.similarity),
                scores=match.scores or None,
                confidence=match.confidence,
                evidence=(
                    {
                        "supporting": [item.to_dict() for item in match.supporting],
                        "conflicting": [item.to_dict() for item in match.conflicting],
                    }
                    if (match.supporting or match.conflicting)
                    else None
                ),
            )
            for match in result.matches
        ]
        FullAnalysisMatch.objects.bulk_create(created_matches, batch_size=batch_size)

        return created_matches

    def build_match_detail(
        self,
        algorithm,
        input_values,
        input_imputed,
        ranked_candidates,
        options,
        nearest_candidates=None,
    ):
        """Add evidence and confidence data to the leading matches."""
        if not ranked_candidates:
            return {}

        nearest_candidates = nearest_candidates or ranked_candidates
        reference_ids = {
            reference_sample_id
            for _, reference_sample_id in (
                list(ranked_candidates) + list(nearest_candidates)
            )
        }
        reference_samples = (
            ReferenceSample.objects
            .filter(id__in=reference_ids)
            .select_related("reference_deposit")
            .prefetch_related("measurements__element")
        )
        by_id = {sample.id: sample for sample in reference_samples}

        # Which deposits the nearest references belong to, in rank order. This
        # is what lets an algorithm judge a match by the company it keeps.
        nearest_deposits = []
        for _, reference_sample_id in nearest_candidates:
            reference_sample = by_id.get(reference_sample_id)
            deposit = reference_sample.reference_deposit if reference_sample else None
            nearest_deposits.append(
                (deposit.three_char_code or deposit.name) if deposit else ""
            )

        detail = {}

        for _, reference_sample_id in ranked_candidates:
            reference_sample = by_id.get(reference_sample_id)
            if reference_sample is None:
                continue

            deposit = reference_sample.reference_deposit
            deposit_id = (
                (deposit.three_char_code or deposit.name)
                if deposit
                else ""
            )

            reference_values, reference_imputed = extract_values(
                reference_sample.measurements.all(),
                options,
            )
            common_elements = set(input_values) & set(reference_values)
            if not common_elements:
                continue

            prepared = prepare_vectors(
                input_values,
                reference_values,
                common_elements,
                options,
                input_imputed | reference_imputed,
            )
            supporting, conflicting = algorithm.evidence(prepared)
            raw_scores = algorithm.raw_scores(prepared)
            confidence = algorithm.confidence(deposit_id, nearest_deposits)

            row = {}
            if raw_scores:
                row["scores"] = raw_scores
            if confidence:
                row["confidence"] = confidence
            if supporting or conflicting:
                row["evidence"] = {
                    "supporting": [item.to_dict() for item in supporting],
                    "conflicting": [item.to_dict() for item in conflicting],
                }

            if row:
                detail[reference_sample_id] = row

        return detail

    def calculate_similarity_score(
        self,
        input_values,
        reference_values,
        common_elements,
        similarity_method=None,
        preprocessing=None,
        imputed_elements=None,
    ):
        """Score one analysed sample against one reference sample."""
        algorithm = get_algorithm(similarity_method)

        return algorithm.score_pair(
            input_values,
            reference_values,
            common_elements,
            preprocessing,
            imputed_elements,
        )


class FullAnalysisResultView(APIView):
    """Return the saved overview for one full analysis."""

    def serialize_input_measurement(self, measurement):
        return {
            "element_symbol": measurement.element.symbol,
            "value": measurement.value,
            "unit": measurement.unit,
            "below_detection_limit": measurement.below_detection_limit,
            "detection_limit": measurement.detection_limit,
        }

    def get_saved_samples(self, full_analysis):
        sample_data = _as_json_object(full_analysis.sample_data)
        saved_samples = sample_data.get("samples")
        if isinstance(saved_samples, list) and saved_samples:
            return saved_samples

        input_measurements = (
            full_analysis.input_measurements
            .select_related("element")
            .all()
        )
        return [{
            **sample_data,
            "sample_code": full_analysis.uploaded_sample_code,
            "name": full_analysis.name,
            "source_filename": full_analysis.source_filename,
            "measurements": [
                self.serialize_input_measurement(measurement)
                for measurement in input_measurements
            ],
        }]

    @extend_schema(
        operation_id="full_analysis_detail",
        responses={200: OpenApiTypes.OBJECT},
    )
    def get(self, request, full_analysis_id):
        try:
            full_analysis = FullAnalysis.objects.get(id=full_analysis_id)
        except FullAnalysis.DoesNotExist:
            return Response(
                {"error": f"Full analysis {full_analysis_id} was not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        saved_samples = self.get_saved_samples(full_analysis)
        match_counts = {
            row["analysed_sample_index"]: row["count"]
            for row in (
                full_analysis.ranked_matches
                .values("analysed_sample_index")
                .annotate(count=Count("id"))
            )
        }
        analysed_samples = [
            {
                "sample_index": sample_index,
                "sample_code": sample.get("sample_code"),
                "name": sample.get("name") or sample.get("sample_code"),
                "latitude": sample.get("latitude"),
                "longitude": sample.get("longitude"),
                "match_count": match_counts.get(sample_index, 0),
            }
            for sample_index, sample in enumerate(saved_samples)
        ]

        # This endpoint stays lightweight even when every sample has thousands
        # of matches. Sample measurements and matches have their own endpoint.
        return Response({
            "full_analysis_id": full_analysis.id,
            "full_analysis": {
                "id": full_analysis.id,
                "name": full_analysis.name,
                "uploaded_sample_code": full_analysis.uploaded_sample_code,
                "source_filename": full_analysis.source_filename,
                "method": full_analysis.method,
                "parameters": full_analysis.parameters,
                "status": full_analysis.status,
                "created_at": full_analysis.created_at,
                "completed_at": full_analysis.completed_at,
                # Provenance, added alongside the existing keys rather than
                # replacing any of them.
                "algorithm_id": full_analysis.algorithm_id,
                "algorithm_version": full_analysis.algorithm_version,
                "pipeline_version": full_analysis.pipeline_version,
                "reference_library_version": full_analysis.reference_library_version,
                "runtime_ms": full_analysis.runtime_ms,
            },
            "analysed_samples": analysed_samples,
            "sample_results": full_analysis.sample_results,
            "projection": full_analysis.projection,
            "warnings": full_analysis.warnings,
        })


class FullAnalysisSampleResultView(FullAnalysisResultView):
    """Return one analysed sample and a page of its ranked matches."""

    def serialize_ranked_match(self, match):
        """Serialise a ranked match and any optional detail blocks."""
        result = {
            "id": match.reference_sample_id,
            "rank": match.rank,
            "similarity_score": match.similarity_score,
        }

        if match.scores:
            result["scores"] = match.scores
        if match.confidence:
            result["confidence"] = match.confidence
        if match.evidence:
            result["evidence"] = match.evidence

        return result

    @extend_schema(
        operation_id="full_analysis_sample_detail",
        responses={200: OpenApiTypes.OBJECT},
    )
    def get(self, request, full_analysis_id, sample_index):
        try:
            full_analysis = FullAnalysis.objects.get(id=full_analysis_id)
        except FullAnalysis.DoesNotExist:
            return Response(
                {"error": f"Full analysis {full_analysis_id} was not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        saved_samples = self.get_saved_samples(full_analysis)
        if sample_index < 0 or sample_index >= len(saved_samples):
            return Response(
                {"error": f"Sample index {sample_index} was not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        try:
            page = max(1, int(request.query_params.get("page", 1)))
            page_size = max(1, min(int(request.query_params.get("page_size", 50)), 100))
        except (TypeError, ValueError):
            return Response(
                {"error": "page and page_size must be integers."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        matches = full_analysis.ranked_matches.filter(
            analysed_sample_index=sample_index,
        )
        count = matches.count()
        start = (page - 1) * page_size
        page_matches = matches[start:start + page_size]
        sample = saved_samples[sample_index]

        return Response({
            **sample,
            "sample_index": sample_index,
            "ranked_matches": {
                "count": count,
                "page": page,
                "page_size": page_size,
                "total_pages": max(1, ceil(count / page_size)),
                "results": [
                    self.serialize_ranked_match(match)
                    for match in page_matches
                ],
            },
        })


class FullAnalysisSampleMapView(FullAnalysisResultView):
    """Return map coordinates for one sample's ranked references."""

    @extend_schema(
        operation_id="full_analysis_sample_map",
        responses={200: OpenApiTypes.OBJECT},
    )
    def get(self, request, full_analysis_id, sample_index):
        try:
            full_analysis = FullAnalysis.objects.get(id=full_analysis_id)
        except FullAnalysis.DoesNotExist:
            return Response(
                {"error": f"Full analysis {full_analysis_id} was not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        saved_samples = self.get_saved_samples(full_analysis)
        if sample_index < 0 or sample_index >= len(saved_samples):
            return Response(
                {"error": f"Sample index {sample_index} was not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        matches = (
            full_analysis.ranked_matches
            .filter(analysed_sample_index=sample_index)
            .select_related(
                "reference_sample",
                "reference_sample__reference_deposit",
            )
        )
        results = []

        for match in matches:
            reference_sample = match.reference_sample
            if reference_sample is None:
                continue

            deposit = reference_sample.reference_deposit
            latitude = (
                reference_sample.latitude
                if reference_sample.latitude is not None
                else (deposit.latitude if deposit else None)
            )
            longitude = (
                reference_sample.longitude
                if reference_sample.longitude is not None
                else (deposit.longitude if deposit else None)
            )

            if latitude is None or longitude is None:
                continue

            results.append({
                "id": reference_sample.id,
                "sample_code": reference_sample.sample_code,
                "rank": match.rank,
                "similarity_score": match.similarity_score,
                "latitude": latitude,
                "longitude": longitude,
            })

        sample = saved_samples[sample_index]
        return Response({
            "sample_index": sample_index,
            "sample_code": sample.get("sample_code"),
            "count": len(results),
            "results": results,
        })


class FullAnalysisMapView(FullAnalysisResultView):
    """Return each unique reference at its best overall similarity."""

    @extend_schema(
        operation_id="full_analysis_map",
        responses={200: OpenApiTypes.OBJECT},
    )
    def get(self, request, full_analysis_id):
        try:
            full_analysis = FullAnalysis.objects.get(id=full_analysis_id)
        except FullAnalysis.DoesNotExist:
            return Response(
                {"error": f"Full analysis {full_analysis_id} was not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        saved_samples = self.get_saved_samples(full_analysis)
        # Ordering by score means the first row encountered for a reference is
        # its best result across all uploaded/analysed samples.
        best_matches = {}
        for match in (
            full_analysis.ranked_matches
            .exclude(reference_sample_id=None)
            .select_related(
                "reference_sample",
                "reference_sample__reference_deposit",
            )
            .order_by("-similarity_score", "rank", "reference_sample_id")
        ):
            best_matches.setdefault(match.reference_sample_id, match)

        globally_ranked = sorted(
            best_matches.values(),
            key=lambda match: (-match.similarity_score, match.reference_sample_id),
        )
        results = []
        matches_without_coordinates = 0

        for overall_rank, match in enumerate(globally_ranked, start=1):
            reference_sample = match.reference_sample
            deposit = reference_sample.reference_deposit
            latitude = reference_sample.latitude
            longitude = reference_sample.longitude
            if latitude is None and deposit is not None:
                latitude = deposit.latitude
            if longitude is None and deposit is not None:
                longitude = deposit.longitude
            if latitude in (None, "") or longitude in (None, ""):
                matches_without_coordinates += 1
                continue

            analysed_sample = (
                saved_samples[match.analysed_sample_index]
                if match.analysed_sample_index < len(saved_samples)
                else {}
            )

            results.append({
                "id": reference_sample.id,
                "sample_code": reference_sample.sample_code,
                "deposit_name": deposit.name if deposit else None,
                "latitude": latitude,
                "longitude": longitude,
                "overall_rank": overall_rank,
                "overall_similarity_score": match.similarity_score,
                "best_analysed_sample_index": match.analysed_sample_index,
                "best_analysed_sample_rank": match.rank,
                "best_analysed_sample_code": (
                    analysed_sample.get("sample_code")
                    or f"Sample {match.analysed_sample_index + 1}"
                ),
            })

        return Response({
            "full_analysis_id": full_analysis.id,
            "count": len(results),
            "total_ranked_matches": len(globally_ranked),
            "matches_without_coordinates": matches_without_coordinates,
            "score_method": "maximum similarity per reference across all analysed samples",
            "results": results,
        })
