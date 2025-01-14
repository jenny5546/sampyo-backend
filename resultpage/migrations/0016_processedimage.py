# Generated by Django 3.1.3 on 2020-12-03 11:42

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('resultpage', '0015_auto_20201202_1551'),
    ]

    operations = [
        migrations.CreateModel(
            name='ProcessedImage',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('photo', models.FileField(blank=True, null=True, upload_to='processed_image')),
                ('saved_file_name', models.TextField(blank=True, null=True)),
                ('raw_image', models.OneToOneField(null=True, on_delete=django.db.models.deletion.CASCADE, to='resultpage.rawimage')),
            ],
        ),
    ]
