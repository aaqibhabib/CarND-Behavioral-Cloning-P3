import cv2
import csv
import numpy as np
from sklearn.utils import shuffle

def get_lines_from_driving_logs(data_path, skip_header=False):
    """
    Returns the lines from a driving log with base directory `data_path`.
    If the file include headers, pass `skip_header=True`.
    """
    lines = []
    with open(data_path + '/driving_log.csv') as csvFile:
        reader = csv.reader(csvFile)
        if skip_header:
            next(reader, None)
        for line in reader:
            lines.append(line)
    return lines

def parse_image_path(fullPath):
    return '/'.join(fullPath.split('/')[-2:])

def load_image_and_measurement(data_path, image_path, measurement, images, measurements):
    """
    Executes the following steps:
      - Loads the image from `data_path` and `imagPath`.
      - Converts the image from BGR to RGB.
      - Adds the image and `measurement` to `images` and `measurements`.
      - Flips the image vertically.
      - Inverts the sign of the `measurement`.
      - Adds the flipped image and inverted `measurement` to `images` and `measurements`.
    """
    original_image = cv2.imread(data_path + '/' + image_path.strip())
    # import pdb; pdb.set_trace()
    image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    images.append(image)
    measurements.append(measurement)
    # Flipping
    images.append(cv2.flip(image,1))
    measurements.append(measurement*-1.0)

def load_images_and_measurements(data_path, skip_header=False, correction=0.2):
    """
    Loads the images and measurements from the driving logs in the directory `data_path`.
    If the file include headers, pass `skip_header=True`.
    `correction` is the value to add/substract to the measurement to use side cameras.
    Returns a pair `(images, measurements)`
    """
    lines = get_lines_from_driving_logs(data_path, skip_header)
    images = []
    measurements = []

    for line in lines:
        measurement = float(line[3])
        # Center
        load_image_and_measurement(data_path, parse_image_path(line[0]), measurement, images, measurements)
        # Left
        load_image_and_measurement(data_path, parse_image_path(line[1]), measurement + correction, images, measurements)
        # Right
        load_image_and_measurement(data_path, parse_image_path(line[2]), measurement - correction, images, measurements)
    
    images, measurements = shuffle(images, measurements, random_state=10)
    return (np.array(images), np.array(measurements))

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

def train_and_save(model, inputs, outputs, model_file, epochs = 3):
    """
    Train the model `model` using 'mse' lost and 'adam' optimizer for the epochs `epochs`.
    The model is saved at `model_file`
    """
    model.compile(loss='mse', optimizer='adam')
    model.fit(inputs, outputs, validation_split=0.2, shuffle=True, nb_epoch=epochs)
    model.save(model_file)
    print("Model saved at " + model_file)

def create_preprocessing_layers():
    """
    Creates a model with the initial pre-processing layers.
    """
    model = Sequential()
    model.add(Cropping2D(cropping=((70,24), (60,60)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))

    return model

def leNet_model():
    """
    Creates a LeNet model.
    """
    model = create_preprocessing_layers()
    model.add(Convolution2D(6,5,5,activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6,5,5,activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    return model

def nvidia_model():
    """
    Creates nvidea Autonomous Car Group model
    """
    model = create_preprocessing_layers()
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model


print('Loading images')
X_train, y_train = load_images_and_measurements('data', skip_header=True)
# model = leNet_model()
model = nvidia_model()
print('Training model')
train_and_save(model, X_train, y_train, 'models/nVidea_data.h5', epochs=7)
print('The End')