from django.db import migrations


class Migration(migrations.Migration):
    """Join the two independent migrations that were created after 0005."""

    dependencies = [
        ("srs", "0006_fullanalysis_sample_data"),
        ("srs", "0006_remove_fullanalysismatch_explanation_and_more"),
    ]

    operations = []
