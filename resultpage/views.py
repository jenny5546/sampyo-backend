import os, requests, time, json, io, base64
import pyexiv2
from django.core.files.base import ContentFile
from django.core.files.images import ImageFile
from django.core.files import File
from PIL import Image, ImageOps
from django.core import serializers
from django.conf import settings
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from .models import RawImage, ProcessedImage, InputImage, Prediction
from django.http import HttpResponse, HttpResponseNotAllowed, HttpResponseRedirect
from resultpage.utils import decode_base64_file, rotate_image
from resultpage.tensorflow.main import predict_image
from resultpage.tensorflow.brightness import get_brightness
from io import BytesIO


def index(request):
    if request.method == 'GET':
        context = {}
        prediction_list = []

        prediction_set = Prediction.objects.all()
        
        for prediction in prediction_set:
            prediction_info = {}
            prediction_info['id'] = prediction.id
            prediction_info['label'] = prediction.label
            prediction_info['classification'] = prediction.classification
            prediction_info['created_at'] = prediction.created_at.strftime("%Y-%m-%d %H:%M") # .strftime("%Y-%m-%d %H:%M:%S")

            input_file_path = os.path.join(settings.MEDIA_ROOT, "%s" %(prediction.input_image.saved_file_name))

            with open(input_file_path, 'rb') as source:
                input_img_encoded_string = base64.b64encode(source.read())
                prediction_info['input_img']  = input_img_encoded_string.decode('utf8')
            
            prediction_list.append(prediction_info)

        context['predictions'] = prediction_list

        # print('context',context)
        context = json.dumps(context)
        return HttpResponse(status=200, content=context)
        
    else:
        return HttpResponseNotAllowed(['GET'])


# Autocrop handler method
def crop_image(raw_image, file_name):
    raw_file_path = os.path.join(settings.MEDIA_ROOT, "raw_image/%s" %(file_name))
    processed_file_path = os.path.join(settings.MEDIA_ROOT, "processed_image/%s" %(file_name))
    img = Image.open(raw_file_path)
    img = ImageOps.exif_transpose(img)

    width, height = img.size

    f, e = os.path.splitext(processed_file_path)

    print('width',width,height)

    croppedImg = None
    if width > height:
        croppedImg = img.crop((500,650,3500,2350))

    elif width < height and width > 3000: 
        croppedImg = img.crop((650,500,2350,3500))

    else: # 직접 촬영한 이미지인 경우,
        croppedImg = img.crop((650,470,2350,3470))

    # If is in .png format, convert to rgb
    if croppedImg.mode in ("RGBA", "P"):
        croppedImg = croppedImg.convert("RGB")

    # croppedImg = croppedImg.rotate(270, expand=True)

    croppedImg.save(f + '.jpg', "JPEG", quality=50)

    # Save To Processed Images
    new_processed_image = ProcessedImage()
    new_processed_image.photo.name = f + '.jpg'

    processed_file_name, processed_file_format = os.path.splitext(file_name)

    new_processed_image.saved_file_name= processed_file_name + '.jpg'
    new_processed_image.raw_image = raw_image
    new_processed_image.save()

    return new_processed_image


@csrf_exempt
def auto_crop(request):
    if request.method == 'POST':
        local_file_name = request.POST.get('local_file_name')
        raw_image_set = RawImage.objects.filter(local_file_name=local_file_name)

        # Wait until raw_image is processed and saved
        while not raw_image_set:
            raw_image_set = RawImage.objects.filter(local_file_name=local_file_name)

        raw_image = raw_image_set.latest('id')

        file_name = raw_image.saved_file_name
        auto_cropped_img = crop_image(raw_image, file_name)

        # Return Auto Crop Result Image Base64 encoded in JSON Format
        processed_file_path = os.path.join(settings.MEDIA_ROOT, "processed_image/%s" %(auto_cropped_img.saved_file_name))
        
        with open(processed_file_path, 'rb') as source:
            encoded_string = base64.b64encode(source.read())
            context = {'img': encoded_string.decode('utf8')}
            context = json.dumps(context)
            return HttpResponse(status=200, content=context)
    else:
        return HttpResponseNotAllowed(['POST'])

@csrf_exempt
def render_brightness(request):
    if request.method == 'POST':
        encoded_raw_image = request.POST.get('base64_encoded')
        local_file_name = request.POST.get('local_file_name')
        decoded_raw_image, file_name = decode_base64_file(encoded_raw_image)
        new_raw_image = RawImage.objects.create(photo=decoded_raw_image, local_file_name=local_file_name, saved_file_name=file_name)
        
        image_file_path = os.path.join(settings.MEDIA_ROOT, "raw_image/%s" %(file_name))
        brightness = get_brightness(image_file_path)

        new_raw_image.brightness = brightness
        new_raw_image.save()

        context = { 'brightness': brightness }
        context = json.dumps(context)
        return HttpResponse(status=200, content=context)

    else:
        return HttpResponseNotAllowed(['POST'])

@csrf_exempt
def render_prediction(request):
    if request.method == 'POST':
        encoded_raw_image = request.POST.get('base64_encoded')
        local_file_name = request.POST.get('local_file_name')
        decoded_raw_image, file_name = decode_base64_file(encoded_raw_image)
        new_input_image = InputImage.objects.create(photo=decoded_raw_image, saved_file_name=file_name)

        image_file_path = os.path.join(settings.MEDIA_ROOT, "input_image/%s" %(file_name))
        weight_file_path = os.path.join(settings.MEDIA_ROOT, "weights/%s" %('2_largfac.hdf5'))

        # Rotate Image Before predicting 
        original_photo = BytesIO(new_input_image.photo.read())
        rotated_photo = BytesIO()

        image = Image.open(original_photo)
        width, height = image.size

        if width < height: # 세로가 더 길면 rotate 시켜준다.
            image = image.rotate(90, expand=True)

        image.save(rotated_photo, 'JPEG')

        f, e = os.path.splitext(image_file_path)

        new_input_image.photo.save(f + '_rotated.jpg', ContentFile(rotated_photo.getvalue()))
        new_input_image.saved_file_name = f + '_rotated.jpg'
        new_input_image.save()

        rotated_image_path = f + '_rotated.jpg'

        os.remove(image_file_path) 

        prediction_result = predict_image(rotated_image_path,weight_file_path)
        prediction_result = prediction_result.item() # convert numpy.int64 -> python int

        new_prediction = Prediction.objects.create(input_image = new_input_image, classification=prediction_result)

        context = { 'classification': prediction_result, 'result_id': new_prediction.id }
        context = json.dumps(context)
        return HttpResponse(status=200, content=context)

    else:
        return HttpResponseNotAllowed(['POST'])

@csrf_exempt
def add_label(request):
    if request.method == 'POST':
        prediction_id = request.POST.get('prediction_id')
        label_content = request.POST.get('label')

        prediction = Prediction.objects.get(id=prediction_id)
        prediction.label = label_content
        prediction.save()

        return HttpResponse(status=200)

    else:
        return HttpResponseNotAllowed(['POST'])


@csrf_exempt
def delete(request):
    if request.method == 'POST':
        prediction_id = request.POST.get('prediction_id')
        prediction = Prediction.objects.get(id=prediction_id)
        prediction.delete()
        return HttpResponse(status=200)

    else:
        return HttpResponseNotAllowed(['POST'])



