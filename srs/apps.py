"""Django application configuration for SRS."""

from django.apps import AppConfig


class SrsConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "srs"
    verbose_name = "Signature Reference System"
