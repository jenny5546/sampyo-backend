# Generated by Django 3.1.3 on 2020-12-03 12:15

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('resultpage', '0016_processedimage'),
    ]

    operations = [
        migrations.AlterField(
            model_name='processedimage',
            name='photo',
            field=models.ImageField(blank=True, null=True, upload_to='processed_image'),
        ),
    ]