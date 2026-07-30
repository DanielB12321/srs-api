from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("srs", "0008_fullanalysismatch_analysed_sample_index"),
    ]

    operations = [
        migrations.DeleteModel(
            name="SimilarityResult",
        ),
        migrations.DeleteModel(
            name="AnalysisRun",
        ),
    ]
