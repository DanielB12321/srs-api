from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .api_views import (
    ReferenceImportViewSet,
    DatasetViewSet,
    ReferenceLibrarySearchView,
    BulkReferenceSampleDetailView,
    FullAnalysisListCreateView,
    FullAnalysisResultView,
    FullAnalysisSampleResultView,
    FullAnalysisSampleMapView,
)
from .views import (
    DepositClassificationViewSet,
    ElementViewSet,
    MineralViewSet,
    ReferenceDepositViewSet,
    ReferenceSampleMeasurementViewSet,
    ReferenceSampleViewSet,
    SampleMeasurementViewSet,
    SampleViewSet,
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

urlpatterns = [
    # Keep this before router.urls so "bulk-details" is not interpreted as a
    # ReferenceSample primary key by the router's detail route.
    path("reference-samples/bulk-details/", BulkReferenceSampleDetailView.as_view()),
    path("", include(router.urls)),
    path("reference-library/search/", ReferenceLibrarySearchView.as_view()),

    path("full-analysis/", FullAnalysisListCreateView.as_view()),
    path("full-analysis/<int:full_analysis_id>/", FullAnalysisResultView.as_view()),
    path(
        "full-analysis/<int:full_analysis_id>/samples/<int:sample_index>/",
        FullAnalysisSampleResultView.as_view(),
    ),
    path(
        "full-analysis/<int:full_analysis_id>/samples/<int:sample_index>/map/",
        FullAnalysisSampleMapView.as_view(),
    ),
]
