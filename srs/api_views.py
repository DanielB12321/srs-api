from math import ceil
from django.db import transaction
from django.db.models import Count
from django.utils import timezone
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

import hashlib
import heapq
import threading
import json

from time import perf_counter

from django.db import close_old_connections, transaction
from django.utils import timezone
from drf_spectacular.utils import extend_schema, OpenApiParameter
from drf_spectacular.types import OpenApiTypes
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework.response import Response
from rest_framework.views import APIView
from .algorithms import available_algorithms, get_algorithm
from .algorithms.base import PairwiseSimilarity
from .algorithms.knn_aitchison import DEFAULT_DETAIL_TOP_N
from .projections import fit_pca, project_points
from .importers import run_import, run_dataset_import
from .preprocessing import (
    PIPELINE_VERSION,
    extract_values,
    normalise_symbol,
    prepare_vectors,
    resolve_options,
)
from .services.units import concentration_to_ppm

from math import ceil


from .models import (
    ReferenceImport,
    Dataset,
    Sample,
    SampleMeasurement,
    ReferenceSample,
    ReferenceSampleMeasurement,
    Element,
    FullAnalysis,
    FullAnalysisInputMeasurement,
    FullAnalysisMatch,
)
from .serializers import (
    SampleSerializer,
    SampleMeasurementSerializer,
    ReferenceSampleSerializer,
    ReferenceSampleMeasurementSerializer,
    ReferenceImportSerializer,
    ReferenceImportUploadSerializer,
    DatasetSerializer,
    DatasetUploadSerializer,
    ReferenceLibrarySearchResultSerializer,
)

def reference_library_version() -> str:
    """
    Identify the reference library a run was scored against.

    The most recent completed import is the anchor. There is no separate
    version field on the library, and inventing one would mean maintaining a
    scheme that could drift from the imports it describes.
    """
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
    hasher = hashlib.sha256()
    for chunk in uploaded_file.chunks():
        hasher.update(chunk)
    uploaded_file.seek(0)
    return hasher.hexdigest()

class ReferenceImportViewSet(viewsets.ModelViewSet):

    # POST   /api/srs/reference-imports/          -> upload data + metadata workbooks
    # GET    /api/srs/reference-imports/          -> list imports (status, counts, timestamps)
    # GET    /api/srs/reference-imports/{id}/     -> single import detail + errors
    # DELETE /api/srs/reference-imports/{id}/     -> remove an import and its derived rows (CASCADE)

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

        return Response(ReferenceImportSerializer(import_row).data, status=status.HTTP_202_ACCEPTED)


class DatasetViewSet(viewsets.ModelViewSet):

    # POST   /api/srs/datasets/          -> upload a dataset CSV
    # GET    /api/srs/datasets/          -> list datasets (status, counts, timestamps)
    # GET    /api/srs/datasets/{id}/     -> single dataset detail + errors
    # DELETE /api/srs/datasets/{id}/     -> remove a dataset and its derived rows (CASCADE)

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

        existing = Dataset.objects.filter(file_sha256=file_hash).first()

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
            )

        try:
            run_dataset_import(dataset.id)
        except Exception:
            dataset.refresh_from_db()
            return Response(
                DatasetSerializer(dataset).data,
                status=status.HTTP_400_BAD_REQUEST,
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
            dataset.refresh_from_db()
            return Response(
                DatasetSerializer(dataset).data,
                status=status.HTTP_400_BAD_REQUEST,
            )

        dataset.refresh_from_db()
        return Response(DatasetSerializer(dataset).data)
    
    @action(detail=True, methods=["get"], url_path="data")
    def data(self, request, pk=None):
        dataset = self.get_object()

        try:
            page = int(request.query_params.get("page", 1))
            page_size = int(request.query_params.get("page_size", 50))
        except ValueError:
            return Response(
                {"error": "page and page_size must be integers."},
                status=400,
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
        total_pages = max(1, ceil(total_rows / page_size))

        start = (page - 1) * page_size
        end = start + page_size
        page_samples = list(samples_qs[start:end])

        all_measurements = (
            SampleMeasurement.objects
            .filter(sample__dataset=dataset)
            .select_related("element")
            .order_by("element__symbol", "unit")
        )

        measurement_columns = []

        for measurement in all_measurements:
            column_name = self._measurement_column_name(measurement)

            if column_name not in measurement_columns:
                measurement_columns.append(column_name)

        columns = ["sample_id", "latitude", "longitude"] + measurement_columns

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
                column_name = self._measurement_column_name(measurement)
                row[column_name] = measurement.value
                row["_measurement_ids"][column_name] = measurement.id

            rows.append(row)

        null_counts = {}

        for column in columns:
            null_counts[column] = sum(
                1
                for row in rows
                if row.get(column) is None or row.get(column) == ""
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


# Reference Library Search query APIView
@extend_schema(
    parameters=[
        OpenApiParameter("deposit_name",      OpenApiTypes.STR,   description="Partial deposit name match. e.g. Olympic"),
        OpenApiParameter("deposit_type",      OpenApiTypes.STR,   description="Exact deposit type. e.g. IOCG, Epithermal, Carlin Au, Granite Related"),
        OpenApiParameter("mineral_system",    OpenApiTypes.STR,   description="Exact mineral system. e.g. Orogenic Au, High Sulphidation Epithermal, Greisen"),
        OpenApiParameter("country",           OpenApiTypes.STR,   description="Partial country name match. e.g. Australia"),
        OpenApiParameter("state_region",      OpenApiTypes.STR,   description="Partial state or region match. e.g. Queensland"),
        OpenApiParameter("sample_code",       OpenApiTypes.STR,   description="Partial sample code match. e.g. 700001"),
        OpenApiParameter("sample_type",       OpenApiTypes.STR,   description="Exact sample type. e.g. VHMS, Epithermal, Skarn / Skarn Au, Porphyry / Porphyry Cu"),
        OpenApiParameter("ore_minerals",      OpenApiTypes.STR,   description="Comma-separated ore minerals (AND logic — sample must contain all). e.g. Pyrite,Chalcopyrite"),
        OpenApiParameter("element",           OpenApiTypes.STR,   description="Element symbol to filter measurements against. e.g. Au, Ag, Cu, Zn"),
        OpenApiParameter("min_value",         OpenApiTypes.FLOAT, description="Minimum measurement value for the specified element. e.g. 1.0"),
        OpenApiParameter("max_value",         OpenApiTypes.FLOAT, description="Maximum measurement value for the specified element. e.g. 10.0"),
        OpenApiParameter("analytical_method", OpenApiTypes.STR,   description="Analytical method used for measurement. e.g. FA (fire assay), AR (aqua regia)"),
        OpenApiParameter("exclude_bdl",       OpenApiTypes.BOOL,  description="Set true to exclude below-detection-limit measurements from element filtering"),
        OpenApiParameter("limit",             OpenApiTypes.INT,   description="Number of results to return. Default 25, max 500"),
        OpenApiParameter("offset",            OpenApiTypes.INT,   description="Number of results to skip for pagination. Default 0"),
    ],
)
class ReferenceLibrarySearchView(APIView):
    def get(self, request):
        qs = ReferenceSample.objects.select_related("reference_deposit").prefetch_related("measurements__element")

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
                # JSON array element search — __contains is PostgreSQL-only,
                # so search the serialised JSON text for the quoted string instead.
                qs = qs.filter(metadata__icontains=f'"{mineral.strip()}"')

        # Measurement-level filters
        if element := p.get("element"):
            f = {"measurements__element__symbol__iexact": element}
            if min_val := p.get("min_value"):
                f["measurements__value__gte"] = float(min_val)
            if max_val := p.get("max_value"):
                f["measurements__value__lte"] = float(max_val)
            if method := p.get("analytical_method"):
                f["measurements__analytical_method__iexact"] = method
            if p.get("exclude_bdl", "").lower() == "true":
                f["measurements__below_detection_limit"] = False
            qs = qs.filter(**f)

        qs = qs.distinct()

        try:
            limit  = max(1, min(int(p.get("limit", 25)), 500))
            offset = max(0, int(p.get("offset", 0)))
        except ValueError:
            return Response({"error": "limit and offset must be integers."}, status=400)

        total = qs.count()
        samples = qs[offset : offset + limit]

        return Response({
            "total": total,
            "limit": limit,
            "offset": offset,
            "results": ReferenceLibrarySearchResultSerializer(samples, many=True).data
        })
    
class BulkReferenceSampleDetailView(APIView):
    """
    POST /api/reference-samples/bulk-details/

    Return complete reference data for up to 50 sample IDs in one response.
    select_related and prefetch_related keep this to a small, fixed number of
    database queries instead of querying once per ranked match.
    """

    def post(self, request):
        requested_ids = request.data.get("ids")

        if not isinstance(requested_ids, list) or not requested_ids:
            return Response(
                {"error": "ids must be a non-empty list."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            # dict.fromkeys removes duplicates without changing requested order.
            sample_ids = list(dict.fromkeys(int(sample_id) for sample_id in requested_ids))
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
    """
    GET /api/algorithms/

    List the similarity algorithms this deployment can run, so the analysis
    setup page can offer the choice instead of hardcoding a list that drifts
    from what the service actually supports. The id of any entry is a valid
    similarity_method for POST /api/full-analysis/.
    """

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
    """
    GET  /api/full-analysis/
    POST /api/full-analysis/

    A POST request queues the complete workflow:
    1. Save the uploaded/test samples and analysis configuration.
    2. Compare each sample with the reference library in background batches.
    3. Save each sample's final highest-scoring reference IDs.

    A GET request returns summaries of previously saved analyses.
    """

    def normalise_element_symbol(self, symbol):
        """Return a consistently capitalised symbol, for example 'CU' -> 'Cu'."""
        # One implementation, in the preprocessing package, so a symbol written
        # by the request path and one read by the pipeline cannot disagree.
        return normalise_symbol(symbol)


    def serialize_full_analysis_summary(self, full_analysis):
        sample_data = full_analysis.sample_data or {}

        if isinstance(sample_data, str):
            try:
                sample_data = json.loads(sample_data)
            except (json.JSONDecodeError, TypeError):
                sample_data = {}

        parameters = full_analysis.parameters or {}

        if isinstance(parameters, str):
            try:
                parameters = json.loads(parameters)
            except (json.JSONDecodeError, TypeError):
                parameters = {}

        samples = sample_data.get("samples") or []

        return {
            "id": full_analysis.id,
            "name": full_analysis.name,
            "uploaded_sample_code": full_analysis.uploaded_sample_code,
            "source_filename": full_analysis.source_filename,
            "method": full_analysis.method,
            "status": full_analysis.status,
            "created_at": full_analysis.created_at,
            "completed_at": full_analysis.completed_at,
            "match_count": full_analysis.ranked_matches.count(),
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

    def get(self, request):
        full_analyses = FullAnalysis.objects.all().order_by("-created_at")

        return Response({
            "count": full_analyses.count(),
            "results": [
                self.serialize_full_analysis_summary(full_analysis)
                for full_analysis in full_analyses
            ],
        })

    def post(self, request):
        analysis_name = request.data.get("analysis_name") or request.data.get("name") or "Uploaded analysis"
        source_filename = request.data.get("source_filename", "")
        similarity_method = (
            request.data.get("similarity_method")
            or "log_difference_similarity"
        )
        preprocessing = request.data.get("preprocessing") or {}

        try:
            top_n = int(request.data.get("top_n", 200))
        except (TypeError, ValueError):
            return Response(
                {"error": "top_n must be an integer."},
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

        # Record which algorithm actually ran, not just what was asked for. An
        # unrecognised similarity_method resolves to the default, and without
        # this a stored analysis would claim to have used something it did not.
        resolved_algorithm = get_algorithm(similarity_method)

        full_analysis = FullAnalysis.objects.create(
            name=analysis_name,
            uploaded_sample_code=first_sample["sample_code"],
            source_filename=source_filename,
            sample_data=sample_snapshot,
            method=similarity_method,
            parameters={
                "top_n": top_n,
                "batch_size": 250,
                "dataset_id": request.data.get("dataset_id"),
                "dataset_name": request.data.get("dataset_name"),
                "selected_elements": request.data.get("selected_elements") or [],
                "preprocessing": preprocessing,
                "similarity_method": similarity_method,
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
            samples = full_analysis.sample_data.get("samples") or []
            parameters = dict(full_analysis.parameters)
            top_n = int(parameters["top_n"])
            batch_size = int(parameters.get("batch_size", 250))
            similarity_method = parameters.get(
                "similarity_method",
                "log_difference_similarity",
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
            FullAnalysis.objects.filter(id=full_analysis_id).update(
                status=FullAnalysis.STATUS_FAILED,
                completed_at=timezone.now(),
                parameters={
                    **(
                        FullAnalysis.objects
                        .filter(id=full_analysis_id)
                        .values_list("parameters", flat=True)
                        .first()
                        or {}
                    ),
                    "error": str(error),
                },
            )
        finally:
            close_old_connections()

    def build_sample_results(self, full_analysis, samples):
        """
        Summarise each analysed sample and its leading matches.

        This is the envelope's per-sample block. It is small by design, holding
        only the top few matches per sample, so the map and overview views can
        be drawn without paging through every ranked match.
        """
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
        """
        Place the analysed samples and their matches on a two-dimensional map.

        The axes are fitted on the matched references rather than on the whole
        library, which bounds both the memory used and the number of points a
        client has to draw. PCA is used because it yields a fixed linear map,
        so an uploaded sample lands in the same space as the references without
        anything being recomputed.

        Returns None when there is too little shared chemistry to place points
        honestly; the projection block is optional precisely so that this is a
        valid outcome.
        """
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
                "below_detection_limit": bool(item.get("below_detection_limit", False)),
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
        """
        Compare one saved test sample with the reference library.

        Only the reference sample ID, rank, and final score are persisted for a
        match. Descriptive reference data stays in the reference library and is
        requested separately when the results page needs it.
        """
        # Resolved once, outside the reference loop, so policy names are not
        # revalidated on every one of a thousand comparisons.
        options = resolve_options(
            preprocessing,
            (full_analysis.parameters or {}).get("selected_elements"),
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

            parameters = dict(
                FullAnalysis.objects
                .filter(id=full_analysis.id)
                .values_list("parameters", flat=True)
                .first()
                or {}
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
        detail_top_n = int(
            (full_analysis.parameters or {}).get("detail_top_n", DEFAULT_DETAIL_TOP_N)
        )
        detail = self.build_match_detail(
            algorithm,
            input_values,
            input_imputed,
            ranked_candidates[:detail_top_n],
            options,
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
        """
        Run an algorithm that needs every reference in memory at once.

        Anything that fits a model or builds a projection cannot be scored one
        reference at a time, so the library is materialised and handed over
        whole. That costs more memory than the streaming path, which is why the
        streaming path remains the default and this is used only when an
        algorithm genuinely requires it.
        """
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

        result = algorithm.compare(
            [{
                "sample_code": full_analysis.uploaded_sample_code,
                "values": input_values,
                "imputed": input_imputed,
            }],
            references,
            {"top_n": top_n, "preprocessing": options},
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
    ):
        """
        Compute the extra per-match blocks for the leading matches only.

        Ranking already used every reference, so nothing is lost from the
        ordering. What is avoided is attaching an evidence block of roughly
        3.4 KB to each of a few hundred rows that nobody opens.

        An algorithm that offers none of these returns empty defaults from the
        base class, so this costs one query and nothing else for the
        concentration-based methods.
        """
        if not ranked_candidates:
            return {}

        reference_ids = [
            reference_sample_id
            for _, reference_sample_id in ranked_candidates
        ]
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
        for _, reference_sample_id in ranked_candidates:
            reference_sample = by_id.get(reference_sample_id)
            deposit = reference_sample.reference_deposit if reference_sample else None
            nearest_deposits.append(
                (deposit.three_char_code or deposit.name) if deposit else ""
            )

        detail = {}

        for (_, reference_sample_id), deposit_id in zip(
            ranked_candidates, nearest_deposits
        ):
            reference_sample = by_id.get(reference_sample_id)
            if reference_sample is None:
                continue

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
        """
        Score one tested sample against one reference sample.

        The arithmetic lives in srs/algorithms/, one module per algorithm. This
        resolves the requested method through the registry and delegates, so an
        unknown or missing name still falls back to the configured default.
        """
        algorithm = get_algorithm(similarity_method)

        return algorithm.score_pair(
            input_values,
            reference_values,
            common_elements,
            preprocessing,
            imputed_elements,
        )


class FullAnalysisResultView(APIView):
    """
    GET /api/full-analysis/<full_analysis_id>/

    Returns one saved full analysis result for the results page.

    The analysed sample includes all saved input measurements. Ranked matches
    are deliberately compact: each contains only the reference sample ID, rank,
    and similarity score. Reference details have their own API endpoints.
    """

    def serialize_input_measurement(self, measurement):
        return {
            "element_symbol": measurement.element.symbol,
            "value": measurement.value,
            "unit": measurement.unit,
            "below_detection_limit": measurement.below_detection_limit,
            "detection_limit": measurement.detection_limit,
        }

    def get_saved_samples(self, full_analysis):
        saved_samples = full_analysis.sample_data.get("samples")
        if isinstance(saved_samples, list) and saved_samples:
            return saved_samples

        input_measurements = (
            full_analysis.input_measurements
            .select_related("element")
            .all()
        )
        return [{
            **full_analysis.sample_data,
            "sample_code": full_analysis.uploaded_sample_code,
            "name": full_analysis.name,
            "source_filename": full_analysis.source_filename,
            "measurements": [
                self.serialize_input_measurement(measurement)
                for measurement in input_measurements
            ],
        }]

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
    """
    GET /api/full-analysis/<analysis_id>/samples/<sample_index>/

    Return one tested sample and one paginated page of its ranked matches.
    """

    def serialize_ranked_match(self, match):
        """
        Serialise one ranked match.

        id, rank, and similarity_score keep the names and positions they have
        always had. The newer blocks are added only when an algorithm actually
        produced them, so a response for a method that produces none is
        byte-identical to what it was before they existed.
        """
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
    """
    GET /api/full-analysis/<analysis_id>/samples/<sample_index>/map/

    Return lightweight coordinates for every ranked reference match belonging
    to one tested sample. Measurements are intentionally excluded.
    """

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
