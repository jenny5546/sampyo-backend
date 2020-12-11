from django.db import models
from django.utils import timezone
import urllib.request
from io import BytesIO
import os
from django.core.files import File

class RawImage(models.Model):
    photo= models.FileField(upload_to='raw_image', blank=True, null=True)
    brightness = models.IntegerField(default=0)
    local_file_name = models.TextField(blank=True, null=True)
    saved_file_name = models.TextField(blank=True, null=True)
    
    def __str__(self):
        return 'saved_file_name=%s, local_file_name=%s' %(self.saved_file_name, self.local_file_name)

class ProcessedImage(models.Model):
    photo = models.FileField(upload_to='processed_image', blank=True, null=True)
    raw_image = models.ForeignKey(RawImage, on_delete=models.CASCADE, null=True)
    saved_file_name = models.TextField(blank=True, null=True)

    def __str__(self):
        return 'saved_file_name=%s' %(self.saved_file_name)

class InputImage(models.Model):
    photo = models.FileField(upload_to='input_image', blank=True, null=True)
    saved_file_name = models.TextField(blank=True, null=True)

class Prediction(models.Model):
    label = models.TextField(default='untitled')
    input_image = models.OneToOneField(InputImage, on_delete=models.CASCADE, null=True)
    classification = models.IntegerField(default=0) # class 0, class 1, class 2 
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(blank=True, null=True)

    def __str__(self):
        return 'id=%d, label=%s, class=%d' %(self.id, self.label, self.classification)
