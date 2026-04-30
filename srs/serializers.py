from rest_framework import serializers

from .models import (
    AnalysisRun,
    Dataset,
    Element,
    ReferenceDeposit,
    ReferenceSample,
    ReferenceSampleMeasurement,
    Sample,
    SampleMeasurement,
    SimilarityResult,
)


class DatasetSerializer(serializers.ModelSerializer):
    class Meta:
        model = Dataset
        fields = "__all__"


class SampleSerializer(serializers.ModelSerializer):
    class Meta:
        model = Sample
        fields = "__all__"


class SampleMeasurementSerializer(serializers.ModelSerializer):
    class Meta:
        model = SampleMeasurement
        fields = "__all__"


class ElementSerializer(serializers.ModelSerializer):
    class Meta:
        model = Element
        fields = "__all__"


class ReferenceDepositSerializer(serializers.ModelSerializer):
    class Meta:
        model = ReferenceDeposit
        fields = "__all__"


class ReferenceSampleSerializer(serializers.ModelSerializer):
    class Meta:
        model = ReferenceSample
        fields = "__all__"


class ReferenceSampleMeasurementSerializer(serializers.ModelSerializer):
    class Meta:
        model = ReferenceSampleMeasurement
        fields = "__all__"


class AnalysisRunSerializer(serializers.ModelSerializer):
    class Meta:
        model = AnalysisRun
        fields = "__all__"


class SimilarityResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = SimilarityResult
        fields = "__all__"