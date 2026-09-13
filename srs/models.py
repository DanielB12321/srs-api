"""Database models for reference data, uploads and saved analyses."""

from django.db import models


# Reference library


class ReferenceImport(models.Model):
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

    source_name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    data_file = models.FileField(upload_to="srs/reference_imports/")
    metadata_file = models.FileField(upload_to="srs/reference_imports/")
    data_sha256 = models.CharField(max_length=255, blank=True)
    metadata_sha256 = models.CharField(max_length=255, blank=True)

    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default=STATUS_PENDING,
    )

    stats = models.JSONField(default=dict, blank=True)
    errors = models.JSONField(default=list, blank=True)

    uploaded_by_id = models.IntegerField(null=True, blank=True)
    uploaded_by_email = models.EmailField(blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"{self.source_name}({self.status})"


class Element(models.Model):
    symbol = models.CharField(max_length=10, unique=True)
    name = models.CharField(max_length=100, blank=True)
    atomic_number = models.PositiveIntegerField(null=True, blank=True)
    default_unit = models.CharField(max_length=50, blank=True)

    def __str__(self):
        return f"{self.symbol}({self.name})"


class Mineral(models.Model):
    code = models.CharField(max_length=50, unique=True)
    name = models.CharField(max_length=100)

    def __str__(self):
        return f"{self.name}({self.code})"


class DepositClassification(models.Model):
    deposit_class = models.CharField(max_length=100, db_column="class")
    sub_class = models.CharField(max_length=100, blank=True, db_column="sub_class")
    notes = models.TextField(blank=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["deposit_class", "sub_class"],
                name="depositclassification_class_subclass_uniq",
            ),
        ]

    def __str__(self):
        if self.sub_class:
            return f"{self.deposit_class} / {self.sub_class}"
        return self.deposit_class


class ReferenceDeposit(models.Model):
    import_ref = models.ForeignKey(
        ReferenceImport,
        on_delete=models.CASCADE,
        related_name="deposits",
        null=True,
        blank=True,
    )
    name = models.CharField(max_length=255)
    three_char_code = models.CharField(max_length=3, db_index=True, default="")
    deposit_type = models.CharField(max_length=100, blank=True)
    mineral_system = models.CharField(max_length=100, blank=True)
    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)
    country = models.CharField(max_length=100, blank=True)
    state_region = models.CharField(max_length=100, blank=True)
    description = models.TextField(blank=True)
    source = models.CharField(max_length=255, blank=True, default="OSNACA")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["name", "three_char_code"],
                name="referencedeposit_name_threechar_uniq",
            ),
        ]

    def __str__(self):
        return f"{self.name} ({self.three_char_code})"


class ReferenceSample(models.Model):
    import_ref = models.ForeignKey(
        ReferenceImport,
        on_delete=models.CASCADE,
        related_name="samples",
        null=True,
        blank=True,
    )
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
    source_dataset = models.CharField(max_length=255, blank=True, default="OSNACA")
    source_reference = models.TextField(blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["import_ref", "sample_code"],
                name="referencesample_import_samplecode_uniq",
            ),
        ]

    def __str__(self):
        return self.sample_code


class ReferenceSampleMeasurement(models.Model):
    import_ref = models.ForeignKey(
        ReferenceImport,
        on_delete=models.CASCADE,
        related_name="measurements",
        null=True,
        blank=True,
    )
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
    analytical_method = models.CharField(max_length=20, blank=True)
    value = models.FloatField(null=True, blank=True)
    unit = models.CharField(max_length=50, blank=True)
    below_detection_limit = models.BooleanField(default=False)
    detection_limit = models.FloatField(null=True, blank=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["reference_sample", "element", "analytical_method"],
                name="refmeasurement_sample_element_method_uniq",
            ),
        ]
        indexes = [
            models.Index(fields=["element", "value"]),
        ]

    def __str__(self):
        method = f" [{self.analytical_method}]" if self.analytical_method else ""
        return f"{self.reference_sample} - {self.element}{method}: {self.value}"


# Uploaded datasets


class Dataset(models.Model):
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

    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)

    uploaded_file = models.FileField(upload_to="srs/datasets/", blank=True, null=True)
    original_filename = models.CharField(max_length=255, blank=True)
    file_sha256 = models.CharField(max_length=64, blank=True, db_index=True)

    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default=STATUS_PENDING,
    )

    row_count = models.PositiveIntegerField(default=0)
    col_count = models.PositiveIntegerField(default=0)
    null_count = models.PositiveIntegerField(default=0)

    stats = models.JSONField(default=dict, blank=True)
    errors = models.JSONField(default=list, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

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


# Saved analyses


class FullAnalysis(models.Model):
    """One saved request to compare uploaded samples with the library."""

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

    name = models.CharField(max_length=255, blank=True)
    uploaded_sample_code = models.CharField(max_length=100, blank=True)
    source_filename = models.CharField(max_length=255, blank=True)
    # Keep a copy of the submitted samples so results can be reopened later.
    sample_data = models.JSONField(default=dict, blank=True)

    method = models.CharField(max_length=100, default="log_difference_similarity")
    parameters = models.JSONField(default=dict, blank=True)

    # Save the method and versions separately so runs are easy to compare.
    # These can be blank for analyses saved before the fields were added.
    algorithm_id = models.CharField(max_length=100, blank=True, default="")
    algorithm_version = models.CharField(max_length=20, blank=True, default="")
    pipeline_version = models.CharField(max_length=20, blank=True, default="")
    reference_library_version = models.CharField(max_length=255, blank=True, default="")
    runtime_ms = models.FloatField(null=True, blank=True)

    # Extra result sections stay blank when the algorithm doesn't provide them.
    sample_results = models.JSONField(null=True, blank=True)
    projection = models.JSONField(null=True, blank=True)
    element_associations = models.JSONField(null=True, blank=True)
    warnings = models.JSONField(default=list, blank=True)

    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default=STATUS_COMPLETED,
    )

    created_by_id = models.IntegerField(null=True, blank=True)
    created_by_email = models.EmailField(blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return self.name or f"Full analysis {self.id}"


class GeochemicalSignature(models.Model):
    """A reusable multi-element profile saved from an analysed sample."""

    KIND_SAMPLE = "sample"
    KIND_COMPOSITE = "composite"
    KIND_CHOICES = [
        (KIND_SAMPLE, "Analysed sample"),
        (KIND_COMPOSITE, "Analysis composite"),
    ]

    full_analysis = models.ForeignKey(
        FullAnalysis,
        on_delete=models.CASCADE,
        related_name="signatures",
    )
    kind = models.CharField(max_length=20, choices=KIND_CHOICES)
    # -1 means the combined profile; 0, 1, 2, etc. refer to individual sample rows.
    sample_index = models.IntegerField(default=-1)
    sample_code = models.CharField(max_length=100, blank=True)
    representative_method = models.CharField(max_length=30)
    value_space = models.CharField(max_length=20, default="ppm")
    elements = models.JSONField(default=list)
    raw_vector = models.JSONField(default=dict)
    vector = models.JSONField(default=dict)
    imputed_elements = models.JSONField(default=list)
    preprocessing = models.JSONField(default=dict)
    pipeline_version = models.CharField(max_length=20, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["sample_index", "kind"]
        constraints = [
            models.UniqueConstraint(
                fields=["full_analysis", "kind", "sample_index"],
                name="unique_analysis_signature",
            ),
        ]

    def __str__(self):
        label = self.sample_code or self.get_kind_display()
        return f"{self.full_analysis} - {label} signature"


class FullAnalysisInputMeasurement(models.Model):
    """A legacy input reading kept for older single-sample analyses."""

    full_analysis = models.ForeignKey(
        FullAnalysis,
        on_delete=models.CASCADE,
        related_name="input_measurements",
    )
    element = models.ForeignKey(
        Element,
        on_delete=models.CASCADE,
        related_name="full_analysis_input_measurements",
    )
    value = models.FloatField(null=True, blank=True)
    unit = models.CharField(max_length=50, blank=True, default="ppm")
    below_detection_limit = models.BooleanField(default=False)
    detection_limit = models.FloatField(null=True, blank=True)

    class Meta:
        unique_together = ("full_analysis", "element")
        ordering = ["element__symbol"]

    def __str__(self):
        return f"{self.full_analysis} - {self.element}: {self.value}"


class FullAnalysisMatch(models.Model):
    """A ranked reference match and its optional supporting detail."""

    full_analysis = models.ForeignKey(
        FullAnalysis,
        on_delete=models.CASCADE,
        related_name="ranked_matches",
    )
    reference_sample = models.ForeignKey(
        ReferenceSample,
        on_delete=models.SET_NULL,
        related_name="full_analysis_matches",
        null=True,
        blank=True,
    )

    # Which input row this match belongs to, starting from 0.
    # Each row in an uploaded CSV gets its own ranking.
    analysed_sample_index = models.PositiveIntegerField(default=0)
    rank = models.PositiveIntegerField()
    similarity_score = models.FloatField()

    # Optional metrics and nearest-reference confidence for detailed matches.
    scores = models.JSONField(null=True, blank=True)
    confidence = models.JSONField(null=True, blank=True)

    # Evidence is saved for the top matches first, then for others as they are opened.
    evidence = models.JSONField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["rank", "-similarity_score"]
        unique_together = ("full_analysis", "analysed_sample_index", "rank")

    def __str__(self):
        return f"{self.full_analysis} - Rank {self.rank}: {self.similarity_score}"
