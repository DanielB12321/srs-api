from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("srs", "0005_fullanalysis_fullanalysismatch_and_more"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="fullanalysismatch",
            name="explanation",
        ),
        migrations.RemoveField(
            model_name="fullanalysismatch",
            name="elements_used",
        ),
    ]
