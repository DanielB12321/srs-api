from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .views import (
    AnalysisRunViewSet,
    DatasetViewSet,
    ElementViewSet,
    ReferenceDepositViewSet,
    ReferenceSampleMeasurementViewSet,
    ReferenceSampleViewSet,
    SampleMeasurementViewSet,
    SampleViewSet,
    SimilarityResultViewSet,
)

router = DefaultRouter()
router.register("datasets", DatasetViewSet)
router.register("samples", SampleViewSet)
router.register("sample-measurements", SampleMeasurementViewSet)
router.register("elements", ElementViewSet)
router.register("reference-deposits", ReferenceDepositViewSet)
router.register("reference-samples", ReferenceSampleViewSet)
router.register("reference-sample-measurements", ReferenceSampleMeasurementViewSet)
router.register("analysis-runs", AnalysisRunViewSet)
router.register("similarity-results", SimilarityResultViewSet)

urlpatterns = [
    path("", include(router.urls)),
]