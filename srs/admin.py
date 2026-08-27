"""Django admin registrations for shared SRS records."""

from django.contrib import admin

from .models import (
    Dataset,
    DepositClassification,
    Element,
    FullAnalysis,
    FullAnalysisInputMeasurement,
    FullAnalysisMatch,
    Mineral,
    ReferenceDeposit,
    ReferenceImport,
    ReferenceSample,
    ReferenceSampleMeasurement,
    Sample,
    SampleMeasurement,
)


# Keep every shared SRS record available to Django administrators.
admin.site.register(
    [
        Dataset,
        DepositClassification,
        Element,
        FullAnalysis,
        FullAnalysisInputMeasurement,
        FullAnalysisMatch,
        Mineral,
        ReferenceDeposit,
        ReferenceImport,
        ReferenceSample,
        ReferenceSampleMeasurement,
        Sample,
        SampleMeasurement,
    ]
)
