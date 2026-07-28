from math import ceil, log10
from django.db import transaction
from django.utils import timezone
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

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
    AnalysisRunSerializer,
    SimilarityResultSerializer,
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



class FullAnalysisListCreateView(APIView):
    """
    GET  /api/full-analysis/
    POST /api/full-analysis/

    This is separate from the old AnalysisRun and SimilarityResult system.

    A POST request performs the complete workflow synchronously:
    1. Save the uploaded/test sample and its measurements.
    2. Compare it with every reference sample.
    3. Save the highest-scoring reference sample IDs.

    A GET request returns summaries of previously saved analyses.
    """

    def normalise_element_symbol(self, symbol):
        """Return a consistently capitalised symbol, for example 'CU' -> 'Cu'."""
        symbol = str(symbol or "").strip()

        if not symbol:
            return ""

        return symbol[0].upper() + symbol[1:].lower()

    def serialize_full_analysis_summary(self, full_analysis):
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
        # Older clients send "name", while the upload page sends "sample_name".
        sample_name = request.data.get("sample_name") or request.data.get("name") or "Uploaded sample"
        uploaded_sample_code = request.data.get("sample_code") or sample_name
        source_filename = request.data.get("source_filename", "")
        measurements = request.data.get("measurements", [])
        top_n = int(request.data.get("top_n", 10))

        # An analysis without submitted measurements cannot be compared.
        if not isinstance(measurements, list) or len(measurements) == 0:
            return Response(
                {"error": "You must provide a non-empty measurements list."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Protect the API from accidentally receiving a huge requested result set.
        top_n = max(1, min(top_n, 50))

        # These records form one result. If an exception occurs, atomic() rolls
        # everything back rather than leaving a partially saved analysis.
        with transaction.atomic():
            full_analysis = FullAnalysis.objects.create(
                name=sample_name,
                uploaded_sample_code=uploaded_sample_code,
                source_filename=source_filename,
                method="log_difference_similarity",
                parameters={
                    "top_n": top_n,
                    "note": "Temporary similarity method using common element log difference.",
                },
                status=FullAnalysis.STATUS_RUNNING,
            )

            created_measurements = []

            for item in measurements:
                # Symbols become dictionary keys later, so cu, Cu, and CU must
                # all resolve to the same database Element.
                element_symbol = self.normalise_element_symbol(
                    item.get("element_symbol") or item.get("symbol")
                )

                if not element_symbol:
                    continue

                value = item.get("value", None)

                # Preserve rows with missing/invalid values as null. They remain
                # part of the test sample but do not participate in comparison.
                if value in ["", None]:
                    numeric_value = None
                else:
                    try:
                        numeric_value = float(value)
                    except (TypeError, ValueError):
                        numeric_value = None

                element, created = Element.objects.get_or_create(
                    symbol=element_symbol,
                    defaults={"name": element_symbol},
                )

                # Only one value is stored per element. For repeated elements in
                # the request, the last submitted value wins.
                measurement, created = FullAnalysisInputMeasurement.objects.update_or_create(
                    full_analysis=full_analysis,
                    element=element,
                    defaults={
                        "value": numeric_value,
                        "unit": item.get("unit", "ppm"),
                        "below_detection_limit": bool(item.get("below_detection_limit", False)),
                        "detection_limit": item.get("detection_limit", None),
                    },
                )

                created_measurements.append(measurement)

            if len(created_measurements) == 0:
                # Every submitted row lacked an element symbol.
                full_analysis.status = FullAnalysis.STATUS_FAILED
                full_analysis.completed_at = timezone.now()
                full_analysis.save()

                return Response(
                    {"error": "No valid measurements were provided."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # This runs synchronously; status stays "running" until it finishes.
            ranked_matches = self.create_ranked_matches(full_analysis, top_n)

            full_analysis.status = FullAnalysis.STATUS_COMPLETED
            full_analysis.completed_at = timezone.now()
            full_analysis.save()

        return Response(
            {
                "message": "Full analysis created.",
                "full_analysis_id": full_analysis.id,
                "ranked_match_count": len(ranked_matches),
                "results_url": f"/api/full-analysis/{full_analysis.id}/",
            },
            status=status.HTTP_201_CREATED,
        )

    def create_ranked_matches(self, full_analysis, top_n):
        """
        Compare one saved test sample with the reference library.

        Only the reference sample ID, rank, and final score are persisted for a
        match. Descriptive reference data stays in the reference library and is
        requested separately when the results page needs it.
        """
        # Logarithms require positive values. Below-detection-limit readings are
        # excluded because they are not exact measured concentrations.
        input_measurements = (
            full_analysis.input_measurements
            .filter(value__isnull=False, below_detection_limit=False)
            .select_related("element")
        )

        # A symbol-to-value dictionary makes finding shared elements inexpensive.
        input_values = {
            measurement.element.symbol: measurement.value
            for measurement in input_measurements
            if measurement.value is not None and measurement.value > 0
        }

        scored_samples = []

        # Prefetch measurements to avoid a new database query for every sample.
        reference_samples = (
            ReferenceSample.objects
            .select_related("reference_deposit")
            .prefetch_related("measurements__element")
            .all()
        )

        for reference_sample in reference_samples:
            reference_values = {}

            for measurement in reference_sample.measurements.all():
                if (
                    measurement.value is not None
                    and measurement.value > 0
                    and not measurement.below_detection_limit
                ):
                    reference_values[measurement.element.symbol] = measurement.value

            # Compare only elements with usable readings in both samples.
            common_elements = sorted(set(input_values.keys()) & set(reference_values.keys()))

            if not common_elements:
                continue

            score = self.calculate_similarity_score(
                input_values,
                reference_values,
                common_elements,
            )

            scored_samples.append({
                "reference_sample": reference_sample,
                "similarity_score": score,
                "elements_used": common_elements,
            })

        # Higher similarity is better, so the first saved result receives rank 1.
        scored_samples.sort(key=lambda item: item["similarity_score"], reverse=True)

        created_matches = []

        for rank, item in enumerate(scored_samples[:top_n], start=1):
            match = FullAnalysisMatch.objects.create(
                full_analysis=full_analysis,
                reference_sample=item["reference_sample"],
                rank=rank,
                similarity_score=item["similarity_score"],
            )

            created_matches.append(match)

        return created_matches

    def calculate_similarity_score(self, input_values, reference_values, common_elements):
        """
        Return mean logarithmic similarity across the shared elements.

        Identical concentrations score 1 for an element; a tenfold difference
        scores 0.5. Logarithms make proportional differences comparable across
        elements whose concentrations have very different magnitudes.
        """
        element_scores = []

        for element_symbol in common_elements:
            input_value = input_values[element_symbol]
            reference_value = reference_values[element_symbol]

            if input_value <= 0 or reference_value <= 0:
                continue

            # This is the absolute difference in orders of magnitude.
            log_difference = abs(log10(input_value) - log10(reference_value))
            element_score = 1 / (1 + log_difference)

            element_scores.append(element_score)

        if not element_scores:
            return 0

        return sum(element_scores) / len(element_scores)


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

    def get(self, request, full_analysis_id):
        try:
            full_analysis = FullAnalysis.objects.get(id=full_analysis_id)
        except FullAnalysis.DoesNotExist:
            return Response(
                {"error": f"Full analysis {full_analysis_id} was not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        input_measurements = (
            full_analysis.input_measurements
            .select_related("element")
            .all()
        )

        ranked_matches = (
            full_analysis.ranked_matches
            .all()
        )

        # Keep the test sample complete, but do not duplicate reference-library
        # records inside every saved analysis response.
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
            },
            "analysed_sample": {
                "sample_code": full_analysis.uploaded_sample_code,
                "name": full_analysis.name,
                "source_filename": full_analysis.source_filename,
                "measurements": [
                    self.serialize_input_measurement(measurement)
                    for measurement in input_measurements
                ],
            },
            "ranked_matches": [
                {
                    # This is ReferenceSample.id, not FullAnalysisMatch.id. The
                    # results page uses it to request reference details.
                    "id": match.reference_sample_id,
                    "rank": match.rank,
                    "similarity_score": match.similarity_score,
                }
                for match in ranked_matches
            ],
        })
