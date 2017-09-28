## Importing different utilities

import csv
import cv2
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

## Importing different elements from Keras for the network
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

## Empty variables/arrays
lines = []
images = []
measurements = []
augmented_images, augmented_measurements = [], []

## Marking the directory where the files are stored
with open('data/normal_drive/normal_drive_log.csv') as csvfile, \
    open('data/normal_drive/normal_drive_second_course.csv') as csvfile, \
    open('data/normal_drive/reverse_drive.csv') as csvfile:
    reader = csv.reader(csvfile)
    
    for line in reader:
        lines.append(line)

    for row in reader:
        steering_center = float(row[3])
        
        # create adjusted steering measurements for the side camera images
        correction = 0.2 # this is a parameter to tune
        steering_left = steering_center + correction
        steering_right = steering_center - correction
            
        # read in images from center, left and right cameras
        path = 'data/normal_drive/IMG' # fill in the path to your training IMG directory
        img_center1 = process_image(np.asarray(Image.open(path1 + row[0])))
        img_left1 = process_image(np.asarray(Image.open(path1 + row[1])))
        img_right1 = process_image(np.asarray(Image.open(path1 + row[2])))
            
        # add images and angles to data set
        car_images.extend(img_center1, img_left1, img_right1)
        steering_angles.extend(steering_center, steering_left, steering_right)

for line in lines:
    for i in range(3):
        source_path = line[0]
        filename = source_path.split('/')[-1]
        current_path = 'data/normal_drive/IMG/' + filename.split('\\')[-1]
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)

for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement * -1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

## Pipeline for the network

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10)

model.save('model.h5')
