from drf_spectacular.utils import extend_schema_field
from drf_spectacular.types import OpenApiTypes
from rest_framework import serializers


@extend_schema_field(OpenApiTypes.BINARY)
class BinaryFileField(serializers.FileField):
    pass

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
    Dataset,
    Sample,
    SampleMeasurement,
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
    data_file     = BinaryFileField()
    metadata_file = BinaryFileField()

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
        read_only_fields = [
            "original_filename",
            "file_sha256",
            "status",
            "row_count",
            "col_count",
            "null_count",
            "stats",
            "errors",
            "completed_at",
            "created_at",
            "updated_at",
        ]

class DatasetUploadSerializer(serializers.ModelSerializer):
    uploaded_file = serializers.FileField(required=True)

    class Meta:
        model = Dataset
        fields = [
            "name",
            "description",
            "uploaded_file",
            "uploaded_by_id",
            "uploaded_by_email",
        ]

class SampleSerializer(serializers.ModelSerializer):
    class Meta:
        model = Sample
        fields = "__all__"


class SampleMeasurementSerializer(serializers.ModelSerializer):
    class Meta:
        model = SampleMeasurement
        fields = "__all__"



# Reference Library Search Query Result Serializer

class ReferenceSampleMeasurementInlineSerializer(serializers.ModelSerializer):
    element_symbol = serializers.CharField(source="element.symbol", read_only=True)

    class Meta:
        model  = ReferenceSampleMeasurement
        fields = [
            "element_symbol", "analytical_method", "value", "unit",
            "below_detection_limit", "detection_limit",
        ]


class ReferenceLibrarySearchResultSerializer(serializers.ModelSerializer):
    deposit_name   = serializers.CharField(source="reference_deposit.name",           allow_null=True, default=None)
    deposit_type   = serializers.CharField(source="reference_deposit.deposit_type",   allow_null=True, default=None)
    mineral_system = serializers.CharField(source="reference_deposit.mineral_system", allow_null=True, default=None)
    country        = serializers.CharField(source="reference_deposit.country",        allow_null=True, default=None)
    measurements   = ReferenceSampleMeasurementInlineSerializer(many=True, read_only=True)

    class Meta:
        model  = ReferenceSample
        fields = [
            "id", "sample_code", "sample_type", "latitude", "longitude",
            "deposit_name", "deposit_type", "mineral_system", "country",
            "metadata", "measurements",
        ]
