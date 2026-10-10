"""Correct OSNACA major-element values imported as percentages."""

from django.db import migrations
from django.db.models import F


PERCENT_ELEMENT_SYMBOLS = (
    "Al",
    "Ca",
    "Fe",
    "K",
    "Mg",
    "Mn",
    "Na",
    "P",
    "S",
    "Si",
    "Ti",
)
PPM_PER_PERCENT = 10000.0


def correct_osnaca_percent_values(apps, schema_editor):
    measurement = apps.get_model("srs", "ReferenceSampleMeasurement")
    queryset = measurement.objects.filter(
        reference_sample__source_dataset="OSNACA",
        element__symbol__in=PERCENT_ELEMENT_SYMBOLS,
        unit="%",
    )
    queryset.update(
        value=F("value") / PPM_PER_PERCENT,
        detection_limit=F("detection_limit") / PPM_PER_PERCENT,
    )


def restore_raw_osnaca_values(apps, schema_editor):
    measurement = apps.get_model("srs", "ReferenceSampleMeasurement")
    queryset = measurement.objects.filter(
        reference_sample__source_dataset="OSNACA",
        element__symbol__in=PERCENT_ELEMENT_SYMBOLS,
        unit="%",
    )
    queryset.update(
        value=F("value") * PPM_PER_PERCENT,
        detection_limit=F("detection_limit") * PPM_PER_PERCENT,
    )


class Migration(migrations.Migration):

    dependencies = [
        ("srs", "0012_geochemicalsignature"),
    ]

    operations = [
        migrations.RunPython(
            correct_osnaca_percent_values,
            restore_raw_osnaca_values,
        ),
    ]
