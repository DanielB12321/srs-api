from rest_framework import serializers

from .models import (
    #Reference Library
    ReferenceImport,
    Element,
    Mineral,
    DepositClassification,
    ReferenceDeposit,
    ReferenceSample,
    ReferenceSampleMeasurement,
    #Others
    AnalysisRun,
    Dataset,
    ReferenceSample,
    Sample,
    SampleMeasurement,
    SimilarityResult
)
 
#Reference Library Serializers

class ReferenceImportSerializer(serializers.ModelSerializer):
    class Meta:
        model = ReferenceImport
        fields = "__all__"
        read_only_fields = (
            "data_sha256",
            "metadata_sha256",
            "status",
            "stats",
            "errors",
            "created_at",
            "completed_at",
        )


class ReferenceImportUploadSerializer(serializers.ModelSerializer):
    class Meta:
        model = ReferenceImport
        fields = (
            "source_name",
            "description",
            "data_file",
            "metadata_file",
            "uploaded_by_id",
            "uploaded_by_email",
        )

class ElementSerializer(serializers.ModelSerializer):
    class Meta:
        model = Element
        fields = "__all__"


class MineralSerializer(serializers.ModelSerializer):
    class Meta:
        model = Mineral
        fields = "__all__"


class DepositClassificationSerializer(serializers.ModelSerializer):
    class Meta:
        model = DepositClassification
        fields = "__all__"


class ReferenceDepositSerializer(serializers.ModelSerializer):
    sample_count = serializers.IntegerField(
        source="reference_samples.count",
        read_only=True,
    )

    class Meta:
        model = ReferenceDeposit
        fields = "__all__"
        read_only_fields = ("import_ref", "created_at", "updated_at")


class ReferenceSampleSerializer(serializers.ModelSerializer):
    deposit_name = serializers.CharField(
        source="reference_deposit.name",
        read_only=True,
    )
    deposit_three_char_code = serializers.CharField(
        source="reference_deposit.three_char_code",
        read_only=True,
    )

    class Meta:
        model = ReferenceSample
        fields = "__all__"
        read_only_fields = ("import_ref", "created_at")


class ReferenceSampleMeasurementSerializer(serializers.ModelSerializer):
    element_symbol = serializers.CharField(
        source="element.symbol",
        read_only=True,
    )
    sample_code = serializers.CharField(
        source="reference_sample.sample_code",
        read_only=True,
    )

    class Meta:
        model = ReferenceSampleMeasurement
        fields = "__all__"
        read_only_fields = ("import_ref",)

# Others 


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



class AnalysisRunSerializer(serializers.ModelSerializer):
    class Meta:
        model = AnalysisRun
        fields = "__all__"


class SimilarityResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = SimilarityResult
        fields = "__all__"