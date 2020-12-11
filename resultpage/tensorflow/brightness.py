from PIL import Image
from PIL import ImageStat

thr = 20

def get_brightness (image_path):
    try:
        image = Image.open(image_path)
        image_conv = image.convert('L')
        stat = ImageStat.Stat(image_conv)
        return stat.mean[0]
    except IOError:
        print('IOError in image file')
        return 0

def check_brightness (image_path):
    try:
        image = Image.open(image_path)
        image_conv = image.convert('L')
        stat = ImageStat.Stat(image_conv)
        br = stat.mean[0]
        if(br <= thr):
            return False
        else:
            return True
    except IOError:
        print('IOError in image file')
        return False