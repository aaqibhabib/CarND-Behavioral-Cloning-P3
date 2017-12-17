import cv2
import csv
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

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
    # modify split char depending on unix or windows training set
    return '/'.join(fullPath.split('/')[-2:])

def load_image_and_measurement(data_path, image_path, measurement):
    """
      - Loads the image from `data_path` and `image_path`.
      - Converts the image from BGR to RGB. (Simulator provides images as RGB)
      - Flips the image vertically.
      - Inverts the sign of the `measurement`.
      - Return (measurements, images) tuple
    """
    measurements, images = [], []
    original_image = cv2.imread(data_path + '/' + image_path.strip())
    image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    images.append(image)
    measurements.append(measurement)
    # Flipping
    images.append(cv2.flip(image,1))
    measurements.append(measurement*-1.0)
    return measurements, images


def generator(data_path, sample_lines, correction, batch_size):
    num_lines = len(sample_lines)
    for offset in range(0, num_lines, batch_size):
        batch_lines = sample_lines[offset:offset+batch_size]

        images = []
        measurements = []

        for line in batch_lines:
            measurement = float(line[3])
            # Center
            center_measurements, center_images = load_image_and_measurement(data_path, parse_image_path(line[0]), measurement)
            # Left
            left_measurements, left_images = load_image_and_measurement(data_path, parse_image_path(line[1]), measurement + correction)
            # Right
            right_measurements, right_images = load_image_and_measurement(data_path, parse_image_path(line[2]), measurement - correction)

            images = images + center_images + left_images + right_images
            measurements = measurements + center_measurements + left_measurements + right_measurements

            X_train = np.array(images)
            y_train = np.array(measurements)

            yield shuffle(X_train, y_train, random_state=10)
    

def load_images_and_measurements(data_path, skip_header=False, correction=0.2, batch_size = 1000):
    """
    Loads the images and measurements from the driving logs in the directory `data_path`.
    `correction` is the value to add/substract to the measurement to use side cameras.
    Returns `(training_generator, validation_generator), (train_lines, validation_lines)`
    """
    lines = get_lines_from_driving_logs(data_path, skip_header)
    train_lines, validation_lines = train_test_split(lines, test_size=0.2)
    training_generator = generator(data_path, train_lines, correction, batch_size)
    validation_generator = generator(data_path, validation_lines, correction, batch_size)
    
    return (training_generator, validation_generator), (train_lines, validation_lines)
    
    

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

def train_and_save(model, generators, samples, model_file, epochs = 3):
    """
    Train the model `model` using 'mse' lost and 'adam' optimizer for the epochs `epochs`.
    The model is saved at `model_file`
    """
    model.compile(loss='mse', optimizer='adam')
    training_generator, validation_generator = generators
    train_lines, validation_lines = samples
    
    model.fit_generator(training_generator, samples_per_epoch=len(train_lines), 
                        validation_data=validation_generator, nb_val_samples=len(validation_lines), nb_epoch=epochs)
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


def main():
    print('Loading images')
    generators, samples = load_images_and_measurements('data', skip_header=True, batch_size=5000)
    model = nvidia_model()
    print('Training model')
    train_and_save(model, generators, samples, 'models/data.h5', epochs=7)
    print('The End')
    
if __name__ == "__main__":
    main()