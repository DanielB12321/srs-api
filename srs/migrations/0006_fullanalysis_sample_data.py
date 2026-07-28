from django.db import migrations, models


class Migration(migrations.Migration):
    """
    Restore the migration that created sample_data on the deployed database.

    Some databases already record this migration as applied. Fresh databases
    still need it in the migration graph so they create the same schema.
    """

    dependencies = [
        ("srs", "0005_fullanalysis_fullanalysismatch_and_more"),
    ]

    operations = [
        migrations.AddField(
            model_name="fullanalysis",
            name="sample_data",
            field=models.JSONField(blank=True, default=dict),
        ),
    ]
