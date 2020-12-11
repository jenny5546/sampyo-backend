# Generated by Django 3.1.3 on 2020-12-01 05:56

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('resultpage', '0003_auto_20201201_1429'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='rawimage',
            name='photo',
        ),
        migrations.AddField(
            model_name='rawimage',
            name='image_file',
            field=models.ImageField(blank=True, null=True, upload_to='raw_image'),
        ),
        migrations.AddField(
            model_name='rawimage',
            name='photo_url',
            field=models.URLField(blank=True, max_length=400),
        ),
    ]