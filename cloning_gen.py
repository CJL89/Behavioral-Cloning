## Importing different utilities
import csv
import cv2
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

## Importing different elements from Keras for the network
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

## Empty variables/arrays
lines = []
images = []
measurements = []
angles = []
augmented_images, augmented_measurements = [], []

## Marking the directory where the files are stored
with open('data/normal_drive/normal_drive_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    
    for line in reader:
        lines.append(line)

correction = 0.2

# Splitting the training data into validation (20%) and training data(80%).
train_lines, validation_lines = train_test_split(lines, test_size=0.2)

# Generator function
def generator(lines, batch_size=32):
    num_samples = len(lines)
    
    # Loop forever so the generator never terminates
    while 1:
        shuffle(lines)
        
        # Loop that iterates throught the pictures in the directory
        for offset in range(0, num_samples, batch_size):
            batch_samples = lines[offset:offset+batch_size]
            
            for batch_sample in batch_samples:
                
                for i in range(3):
                    name = batch_sample[i]
                    image_ = cv2.imread(name)
                    
                    # Center
                    if i == 0:
                        angle = float(batch_sample[3])
                    elif i == 1:
                        angle = float(batch_sample[3]) + correction
                    elif i == 2:
                        angle = float(batch_sample[3]) - correction
            
                    images.append(image_)
                    angles.append(angle)
                    images.append(cv2.flip(image_,1))
                    angles.append(angle*-1.0)
            
                # Center camera
#                center_image = cv2.imread(name)
#                center_angle = float(batch_sample[3])
#
#                # Appending images to arrays
#                images.append(center_image)
#                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Creating variables that the network will train through
train_generator = generator(train_lines, batch_size=32)
validation_generator = generator(validation_lines, batch_size=32)

# Trimmed image format
ch, row, col = 3, 160, 320

## Pipeline for the network
model = Sequential()

# Preprocessing the input data
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(row, col, ch)))

# Cropping images so the center of the image is only displayed and the neural network can train better
model.add(Cropping2D(cropping=((70, 25), (0, 0))))

# NVIDIA neural network format
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid'))
model.add(Activation('relu'))

# Adding dropout to the convolution
model.add(Dropout(0.5))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))

# Adding dropout to the convolution
model.add(Dropout(0.5))

# Flattening the output
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(train_lines) * 6, validation_data=validation_generator,nb_val_samples=len(validation_lines) * 6, nb_epoch=5)

# Saving model as an .h5 format
model.save('model.h5')
