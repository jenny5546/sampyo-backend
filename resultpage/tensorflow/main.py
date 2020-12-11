from .config import *
from .model import *

## Set TF environment variables
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


## Predict on one image
def predict_image(image_path, weights_path):

    sampyoNet = SampyoNet()
    sampyoNet.load_weights(weights_path)

    image = Image.open(image_path)
    image = image.resize((INPUT_WIDTH, INPUT_HEIGHT))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = sampyoNet.predict(image)
    prediction = np.argmax(prediction)
    return prediction



## Predict on a set of images, specified by a CSV file
def predict_csv(csv_path, weights_path):

    sampyoNet = SampyoNet()
    sampyoNet.load_weights(weights_path)

    prediction = dict()
    with open(csv_path, 'r') as csv:
        csv.readline()
        for record in csv:
            image_path = record.split(',')[0]
            prediction[image_path] = -1

    for image_path in tqdm.tqdm(prediction, desc='evaluating images...'):
        image = Image.open(image_path)
        image = image.resize((INPUT_WIDTH, INPUT_HEIGHT))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        _prediction = sampyoNet.predict(image)
        _prediction = np.argmax(_prediction)
        prediction[image_path] = _prediction

    return prediction
    


## Direct execution from the command line
if __name__ == '__main__':

    help_message = 'python3 main.py -w weights_path.hdf5 -i image_path.jpg OR -a csv_path.csv'

    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv,"i:w:a:")

    for opt, arg in opts:
        if opt in ("-w"):
            weights_path = arg
        elif opt in ("-i"):
            image_path = arg
        elif opt in ("-a"):
            csv_path = arg
    
    if 'weights_path' in locals() and 'image_path' in locals():
        prediction = predict_image(image_path, weights_path)
        print('==================================================================')
        print('Prediction on <{}>, using <{}>'.format(image_path, weights_path))
        print('Prediction : {}'.format(prediction))
        print('==================================================================')

    elif 'weights_path' in locals() and 'csv_path' in locals():
        prediction = predict_csv(csv_path, weights_path)

        input_csv_path = csv_path
        output_csv_path = csv_path.replace('.csv', '_prediction.csv')
        input_csv = open(input_csv_path, 'r')
        output_csv = open(output_csv_path, 'w')
        
        fields = input_csv.readline().replace('\n', '').split(',')
        output_csv.write(','.join(fields) + ',')
        fields = fields[1:]
        output_csv.write(','.join([field + '_pred' for field in fields]) + '\n')

        for record in input_csv:
            output_csv.write(record.replace('\n', ','))
            _prediction = prediction[record.split(',')[0]]
            output_csv.write('{}'.format(_prediction) + '\n')
            
        input_csv.close()
        output_csv.close()

        print('==================================================================')
        print('Prediction on <{}>, using <{}>'.format(csv_path, weights_path))
        print('Prediction saved to <{}>'.format(output_csv_path))
        print('==================================================================')

    else:
        print(help_message)
