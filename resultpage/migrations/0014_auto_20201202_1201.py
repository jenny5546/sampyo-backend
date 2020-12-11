# Generated by Django 3.1.3 on 2020-12-02 03:01

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('resultpage', '0013_auto_20201201_1813'),
    ]

    operations = [
        migrations.CreateModel(
            name='InputImage',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('photo', models.FileField(blank=True, null=True, upload_to='input_image')),
                ('saved_file_name', models.TextField(blank=True, null=True)),
            ],
        ),
        migrations.RemoveField(
            model_name='result',
            name='processed_image',
        ),
        migrations.AddField(
            model_name='processedimage',
            name='raw_image',
            field=models.OneToOneField(null=True, on_delete=django.db.models.deletion.CASCADE, to='resultpage.rawimage'),
        ),
        migrations.AddField(
            model_name='processedimage',
            name='saved_file_name',
            field=models.TextField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='rawimage',
            name='local_file_name',
            field=models.TextField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='rawimage',
            name='saved_file_name',
            field=models.TextField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='result',
            name='classification',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='result',
            name='input_image',
            field=models.OneToOneField(null=True, on_delete=django.db.models.deletion.CASCADE, to='resultpage.inputimage'),
        ),
    ]