# Generated by Django 3.1.3 on 2020-12-01 06:48

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('resultpage', '0007_auto_20201201_1546'),
    ]

    operations = [
        migrations.AlterField(
            model_name='rawimage',
            name='url',
            field=models.URLField(blank=True, max_length=900),
        ),
    ]