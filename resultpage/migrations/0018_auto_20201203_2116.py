# Generated by Django 3.1.3 on 2020-12-03 12:16

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('resultpage', '0017_auto_20201203_2115'),
    ]

    operations = [
        migrations.AlterField(
            model_name='processedimage',
            name='photo',
            field=models.FileField(blank=True, null=True, upload_to='processed_image'),
        ),
    ]