# Generated by Django 3.1.3 on 2020-12-01 05:29

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('resultpage', '0002_auto_20201201_1428'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='result',
            name='raw_image',
        ),
        migrations.AlterField(
            model_name='rawimage',
            name='photo',
            field=models.ImageField(blank=True, null=True, upload_to=''),
        ),
    ]