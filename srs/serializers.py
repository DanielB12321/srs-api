"""REST serializers for SRS models and upload requests."""

from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import extend_schema_field
from rest_framework import serializers

from .models import (
    Dataset,
    DepositClassification,
    Element,
    Mineral,
    ReferenceDeposit,
    ReferenceImport,
    ReferenceSample,
    ReferenceSampleMeasurement,
    Sample,
    SampleMeasurement,
)


@extend_schema_field(OpenApiTypes.BINARY)
class BinaryFileField(serializers.FileField):
    """Describe uploaded files correctly in the generated API schema."""


# Reference library


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
    data_file = BinaryFileField()
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
        read_only_fields = ("uploaded_by_id", "uploaded_by_email")


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
    sample_count = serializers.IntegerField(read_only=True)

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


# Uploaded datasets


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
            "uploaded_by_id",
            "uploaded_by_email",
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
        read_only_fields = ["uploaded_by_id", "uploaded_by_email"]

    def validate_uploaded_file(self, file):
        if not file.name.lower().endswith(".csv"):
            raise serializers.ValidationError(
                "Only CSV files are supported."
            )

        if file.size == 0:
            raise serializers.ValidationError(
                "The uploaded CSV file is empty."
            )

        return file

class SampleSerializer(serializers.ModelSerializer):
    class Meta:
        model = Sample
        fields = "__all__"


class SampleMeasurementSerializer(serializers.ModelSerializer):
    class Meta:
        model = SampleMeasurement
        fields = "__all__"


# Reference-library search results


class ReferenceSampleMeasurementInlineSerializer(serializers.ModelSerializer):
    element_symbol = serializers.CharField(source="element.symbol", read_only=True)

    class Meta:
        model = ReferenceSampleMeasurement
        fields = [
            "element_symbol",
            "analytical_method",
            "value",
            "unit",
            "below_detection_limit",
            "detection_limit",
        ]


class ReferenceLibrarySearchResultSerializer(serializers.ModelSerializer):
    deposit_name = serializers.CharField(
        source="reference_deposit.name",
        read_only=True,
        allow_null=True,
        default=None,
    )
    deposit_type = serializers.CharField(
        source="reference_deposit.deposit_type",
        read_only=True,
        allow_null=True,
        default=None,
    )
    mineral_system = serializers.CharField(
        source="reference_deposit.mineral_system",
        read_only=True,
        allow_null=True,
        default=None,
    )
    country = serializers.CharField(
        source="reference_deposit.country",
        read_only=True,
        allow_null=True,
        default=None,
    )
    measurements = ReferenceSampleMeasurementInlineSerializer(
        many=True,
        read_only=True,
    )

    class Meta:
        model = ReferenceSample
        fields = [
            "id",
            "sample_code",
            "sample_type",
            "latitude",
            "longitude",
            "deposit_name",
            "deposit_type",
            "mineral_system",
            "country",
            "metadata",
            "measurements",
        ]
