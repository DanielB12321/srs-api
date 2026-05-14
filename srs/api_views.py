import hashlib
import threading

from django.db import transaction
from django.utils import timezone
from drf_spectacular.utils import extend_schema, OpenApiParameter
from drf_spectacular.types import OpenApiTypes
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework.response import Response
from rest_framework.views import APIView
from .importers import run_import, run_dataset_import

from math import ceil


from .models import ReferenceImport, Dataset, Sample, SampleMeasurement, ReferenceSample
from .serializers import (
    ReferenceImportSerializer,
    ReferenceImportUploadSerializer,
    DatasetSerializer,
    DatasetUploadSerializer,
    ReferenceLibrarySearchResultSerializer
)

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
    
class SampleLocationsView(APIView):
    def get(self, request):
        samples = (
            ReferenceSample.objects
            .exclude(latitude__isnull=True)
            .exclude(longitude__isnull=True)
            .select_related("reference_deposit")
            .order_by("sample_code")
        )

        data = []

        for sample in samples:
            data.append({
                "id": sample.id,
                "sample_id": sample.sample_code,
                "latitude": sample.latitude,
                "longitude": sample.longitude,
            })

        return Response(data)
