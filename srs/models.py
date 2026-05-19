from django.conf import settings
from django.db import models

# Reference Library Schema 

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
            models.UniqueConstraint(fields=["deposit_class", "sub_class"], name="depositclassification_class_subclass_uniq"),
        ]

    def __str__(self):
        return f"{self.deposit_class} / {self.sub_class}" if self.sub_class else self.deposit_class
    
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

# Pre-existing models below   

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


# class Element(models.Model):
#     symbol = models.CharField(max_length=10, unique=True)
#     name = models.CharField(max_length=100, blank=True)
#     atomic_number = models.PositiveIntegerField(null=True, blank=True)
#     default_unit = models.CharField(max_length=50, blank=True)

#     def __str__(self):
#         return self.symbol


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


# class ReferenceDeposit(models.Model):
#     name = models.CharField(max_length=255)
#     deposit_type = models.CharField(max_length=100, blank=True)
#     mineral_system = models.CharField(max_length=100, blank=True)
#     country = models.CharField(max_length=100, blank=True)
#     state_region = models.CharField(max_length=100, blank=True)
#     description = models.TextField(blank=True)
#     source = models.CharField(max_length=255, blank=True)
#     created_at = models.DateTimeField(auto_now_add=True)
#     updated_at = models.DateTimeField(auto_now=True)

#     def __str__(self):
#         return self.name


# class ReferenceSample(models.Model):
#     reference_deposit = models.ForeignKey(
#         ReferenceDeposit,
#         on_delete=models.CASCADE,
#         related_name="reference_samples",
#         null=True,
#         blank=True,
#     )
#     sample_code = models.CharField(max_length=100)
#     latitude = models.FloatField(null=True, blank=True)
#     longitude = models.FloatField(null=True, blank=True)
#     sample_type = models.CharField(max_length=100, blank=True)
#     source_dataset = models.CharField(max_length=255, blank=True)
#     source_reference = models.TextField(blank=True)
#     metadata = models.JSONField(default=dict, blank=True)
#     created_at = models.DateTimeField(auto_now_add=True)

#     class Meta:
#         unique_together = ("reference_deposit", "sample_code")

#     def __str__(self):
#         return self.sample_code


# class ReferenceSampleMeasurement(models.Model):
#     reference_sample = models.ForeignKey(
#         ReferenceSample,
#         on_delete=models.CASCADE,
#         related_name="measurements",
#     )
#     element = models.ForeignKey(
#         Element,
#         on_delete=models.CASCADE,
#         related_name="reference_measurements",
#     )
#     value = models.FloatField(null=True, blank=True)
#     unit = models.CharField(max_length=50, blank=True)
#     below_detection_limit = models.BooleanField(default=False)
#     detection_limit = models.FloatField(null=True, blank=True)

#     class Meta:
#         unique_together = ("reference_sample", "element")

#     def __str__(self):
#         return f"{self.reference_sample} - {self.element}: {self.value}"


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

        class FullAnalysis(models.Model):
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

    method = models.CharField(max_length=100, default="log_difference_similarity")
    parameters = models.JSONField(default=dict, blank=True)

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

class FullAnalysis(models.Model):
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

    method = models.CharField(max_length=100, default="log_difference_similarity")
    parameters = models.JSONField(default=dict, blank=True)

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


class FullAnalysisInputMeasurement(models.Model):
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

    rank = models.PositiveIntegerField()
    similarity_score = models.FloatField()
    elements_used = models.JSONField(default=list, blank=True)
    explanation = models.JSONField(default=dict, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["rank", "-similarity_score"]
        unique_together = ("full_analysis", "rank")

    def __str__(self):
        return f"{self.full_analysis} - Rank {self.rank}: {self.similarity_score}"