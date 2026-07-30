from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("srs", "0007_merge_full_analysis_migrations"),
    ]

    operations = [
        migrations.AddField(
            model_name="fullanalysismatch",
            name="analysed_sample_index",
            field=models.PositiveIntegerField(default=0),
        ),
        migrations.AlterUniqueTogether(
            name="fullanalysismatch",
            unique_together={
                ("full_analysis", "analysed_sample_index", "rank"),
            },
        ),
    ]
