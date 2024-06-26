# Generated by Django 4.1 on 2024-04-23 06:43

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('predict', '0003_predresults_alcohol_predresults_density_and_more'),
    ]

    operations = [
        migrations.RenameField(
            model_name='predresults',
            old_name='petal_length',
            new_name='citric_acid',
        ),
        migrations.RenameField(
            model_name='predresults',
            old_name='petal_width',
            new_name='fixed_acidity',
        ),
        migrations.RenameField(
            model_name='predresults',
            old_name='sepal_length',
            new_name='regression',
        ),
        migrations.RenameField(
            model_name='predresults',
            old_name='sepal_width',
            new_name='residual_sugar',
        ),
        migrations.RemoveField(
            model_name='predresults',
            name='classification',
        ),
        migrations.AddField(
            model_name='predresults',
            name='volatile_acidity',
            field=models.FloatField(default=0.1234),
            preserve_default=False,
        ),
    ]
