from django.conf import settings
from django.db import models


class Dataset(models.Model):
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    uploaded_file = models.FileField(upload_to="srs/datasets/", blank=True, null=True)

    uploaded_by_id = models.IntegerField(null=True, blank=True)
    uploaded_by_email = models.EmailField(blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name

class Sample(models.Model):
    dataset = models.ForeignKey(
        Dataset,
        on_delete=models.CASCADE,
        related_name="samples",
    )
    sample_code = models.CharField(max_length=100)
    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("dataset", "sample_code")

    def __str__(self):
        return self.sample_code


class Element(models.Model):
    symbol = models.CharField(max_length=10, unique=True)
    name = models.CharField(max_length=100, blank=True)
    atomic_number = models.PositiveIntegerField(null=True, blank=True)
    default_unit = models.CharField(max_length=50, blank=True)

    def __str__(self):
        return self.symbol


class SampleMeasurement(models.Model):
    sample = models.ForeignKey(
        Sample,
        on_delete=models.CASCADE,
        related_name="measurements",
    )
    element = models.ForeignKey(
        Element,
        on_delete=models.CASCADE,
        related_name="sample_measurements",
    )
    value = models.FloatField(null=True, blank=True)
    unit = models.CharField(max_length=50, blank=True)
    below_detection_limit = models.BooleanField(default=False)
    detection_limit = models.FloatField(null=True, blank=True)

    class Meta:
        unique_together = ("sample", "element")

    def __str__(self):
        return f"{self.sample} - {self.element}: {self.value}"


class ReferenceDeposit(models.Model):
    name = models.CharField(max_length=255)
    deposit_type = models.CharField(max_length=100, blank=True)
    mineral_system = models.CharField(max_length=100, blank=True)
    country = models.CharField(max_length=100, blank=True)
    state_region = models.CharField(max_length=100, blank=True)
    description = models.TextField(blank=True)
    source = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name


class ReferenceSample(models.Model):
    reference_deposit = models.ForeignKey(
        ReferenceDeposit,
        on_delete=models.CASCADE,
        related_name="reference_samples",
        null=True,
        blank=True,
    )
    sample_code = models.CharField(max_length=100)
    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)
    sample_type = models.CharField(max_length=100, blank=True)
    source_dataset = models.CharField(max_length=255, blank=True)
    source_reference = models.TextField(blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("reference_deposit", "sample_code")

    def __str__(self):
        return self.sample_code


class ReferenceSampleMeasurement(models.Model):
    reference_sample = models.ForeignKey(
        ReferenceSample,
        on_delete=models.CASCADE,
        related_name="measurements",
    )
    element = models.ForeignKey(
        Element,
        on_delete=models.CASCADE,
        related_name="reference_measurements",
    )
    value = models.FloatField(null=True, blank=True)
    unit = models.CharField(max_length=50, blank=True)
    below_detection_limit = models.BooleanField(default=False)
    detection_limit = models.FloatField(null=True, blank=True)

    class Meta:
        unique_together = ("reference_sample", "element")

    def __str__(self):
        return f"{self.reference_sample} - {self.element}: {self.value}"


class AnalysisRun(models.Model):
    STATUS_PENDING = "pending"
    STATUS_RUNNING = "running"
    STATUS_COMPLETED = "completed"
    STATUS_FAILED = "failed"

    STATUS_CHOICES = [
        (STATUS_PENDING, "Pending"),
        (STATUS_RUNNING, "Running"),
        (STATUS_COMPLETED, "Completed"),
        (STATUS_FAILED, "Failed"),
    ]

    dataset = models.ForeignKey(
        Dataset,
        on_delete=models.CASCADE,
        related_name="analysis_runs",
    )
    name = models.CharField(max_length=255, blank=True)
    created_by_id = models.IntegerField(null=True, blank=True)
    created_by_email = models.EmailField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    method = models.CharField(max_length=100, blank=True)
    comparison_level = models.CharField(max_length=100, blank=True)
    parameters = models.JSONField(default=dict, blank=True)
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default=STATUS_PENDING,
    )

    def __str__(self):
        return self.name or f"Analysis run {self.id}"


class SimilarityResult(models.Model):
    analysis_run = models.ForeignKey(
        AnalysisRun,
        on_delete=models.CASCADE,
        related_name="results",
    )
    sample = models.ForeignKey(
        Sample,
        on_delete=models.CASCADE,
        related_name="similarity_results",
        null=True,
        blank=True,
    )
    reference_sample = models.ForeignKey(
        ReferenceSample,
        on_delete=models.CASCADE,
        related_name="similarity_results",
        null=True,
        blank=True,
    )
    reference_deposit = models.ForeignKey(
        ReferenceDeposit,
        on_delete=models.CASCADE,
        related_name="similarity_results",
        null=True,
        blank=True,
    )
    similarity_score = models.FloatField()
    rank = models.PositiveIntegerField(null=True, blank=True)
    elements_used = models.JSONField(default=list, blank=True)
    explanation = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["rank", "-similarity_score"]

    def __str__(self):
        return f"{self.analysis_run} - {self.similarity_score}"