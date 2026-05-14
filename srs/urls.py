from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .api_views import (
    ReferenceImportViewSet,
    DatasetViewSet,
    ReferenceLibrarySearchView,
    SampleLocationsView,
)
from .views import (
    AnalysisRunViewSet,
    DepositClassificationViewSet,
    ElementViewSet,
    MineralViewSet,
    ReferenceDepositViewSet,
    ReferenceSampleMeasurementViewSet,
    ReferenceSampleViewSet,
    SampleMeasurementViewSet,
    SampleViewSet,
    SimilarityResultViewSet,
)

router = DefaultRouter()
# Reference library
router.register("reference-imports", ReferenceImportViewSet)
router.register("minerals", MineralViewSet)
router.register("deposit-classifications", DepositClassificationViewSet)
router.register("reference-deposits", ReferenceDepositViewSet)
router.register("reference-samples", ReferenceSampleViewSet)
router.register("reference-sample-measurements", ReferenceSampleMeasurementViewSet)
router.register("elements", ElementViewSet)

# Pre-existing
router.register("datasets", DatasetViewSet)
router.register("samples", SampleViewSet)
router.register("sample-measurements", SampleMeasurementViewSet)
router.register("analysis-runs", AnalysisRunViewSet)
router.register("similarity-results", SimilarityResultViewSet)

urlpatterns = [
    path("", include(router.urls)),
    path("reference-library/search/", ReferenceLibrarySearchView.as_view()),
    path("sample-locations/", SampleLocationsView.as_view()),
]
