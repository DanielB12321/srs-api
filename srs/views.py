from rest_framework import viewsets

from .models import (
    AnalysisRun,
    Mineral, 
    DepositClassification,
    Dataset,
    Element,
    ReferenceDeposit,
    ReferenceSample,
    ReferenceSampleMeasurement,
    Sample,
    SampleMeasurement,
    SimilarityResult
)

from .serializers import (
    AnalysisRunSerializer,
    DatasetSerializer,
    ElementSerializer,
    ReferenceDepositSerializer,
    ReferenceSampleMeasurementSerializer,
    ReferenceSampleSerializer,
    SampleMeasurementSerializer,
    SampleSerializer,
    SimilarityResultSerializer,
    MineralSerializer,
    DepositClassificationSerializer
)


class DatasetViewSet(viewsets.ModelViewSet):
    queryset = Dataset.objects.all().order_by("-created_at")
    serializer_class = DatasetSerializer


class SampleViewSet(viewsets.ModelViewSet):
    queryset = Sample.objects.all().order_by("sample_code")
    serializer_class = SampleSerializer


class SampleMeasurementViewSet(viewsets.ModelViewSet):
    queryset = SampleMeasurement.objects.all()
    serializer_class = SampleMeasurementSerializer


class ElementViewSet(viewsets.ModelViewSet):
    queryset = Element.objects.all().order_by("symbol")
    serializer_class = ElementSerializer


class ReferenceDepositViewSet(viewsets.ModelViewSet):
    queryset = ReferenceDeposit.objects.all().order_by("name")
    serializer_class = ReferenceDepositSerializer


class ReferenceSampleViewSet(viewsets.ModelViewSet):
    queryset = ReferenceSample.objects.all().order_by("sample_code")
    serializer_class = ReferenceSampleSerializer


class ReferenceSampleMeasurementViewSet(viewsets.ModelViewSet):
    queryset = ReferenceSampleMeasurement.objects.all()
    serializer_class = ReferenceSampleMeasurementSerializer

    def get_queryset(self):
        queryset = super().get_queryset().select_related("element", "reference_sample")
        reference_sample_id = self.request.query_params.get("reference_sample")
        if reference_sample_id:
            queryset = queryset.filter(reference_sample_id=reference_sample_id)
        return queryset


class AnalysisRunViewSet(viewsets.ModelViewSet):
    queryset = AnalysisRun.objects.all().order_by("-created_at")
    serializer_class = AnalysisRunSerializer


class SimilarityResultViewSet(viewsets.ModelViewSet):
    queryset = SimilarityResult.objects.all().order_by("rank", "-similarity_score")
    serializer_class = SimilarityResultSerializer


class MineralViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Mineral.objects.all().order_by("name")
    serializer_class = MineralSerializer


class DepositClassificationViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = DepositClassification.objects.all().order_by("deposit_class", "sub_class")
    serializer_class = DepositClassificationSerializer
