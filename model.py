from keras.layers import Flatten, Convolution2D, Dense, Lambda, Cropping2D, Dropout, Activation
from keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import csv
import cv2
import sklearn
import numpy as np
import matplotlib.pyplot as plt

## Empty variables/arrays
samples = []

## Marking the directory where the files are stored
with open('data/normal_drive/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)

    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size = 0.2)

# Generator function
def generator(samples, batch_size = 64):
    num_samples = len(samples)

    # Creating infinite loop
    while 1:
        
        # Shuffling the data to prevent biased learning
        sklearn.utils.shuffle(samples)
        
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            
            # Creating empty arrays to append later
            images = []
            angles = []
            for batch_sample in batch_samples:

                # Looping through images
                for i in range(3):
                    
                    # Correction to both left and right pictures
                    correction = 0
                    if i == 1:
                        correction = 0.1
                    elif i == 2:
                        correction = -0.1
                    
                    source_path = batch_sample[i]
                    filename = source_path.split('/')[-1]
                    name = 'data/normal_drive/IMG/' + filename
                    
                    #Uses cv2.imread to read the image
                    image = cv2.imread(name)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                   
                    # Adding correction to pictures
                    angle = float(batch_sample[3]) + correction
            
                # Appending images to arrays
                images.append(image)
                angles.append(angle)
                
                # Inverting images.
                images.append(cv2.flip(image, 1))
                angles.append(angle * -1.0)
            
            # Trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Creating variables that the network will train through
train_generator = generator(train_samples, batch_size = 32)
validation_generator = generator(validation_samples, batch_size = 32)

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
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))

# Compiling the data
model.compile(loss='mse', optimizer='adam')

# Saving the results of the generator in a variable
model.fit_generator(train_generator, samples_per_epoch = len(train_samples) * 6, validation_data = validation_generator, nb_val_samples = len(validation_samples) * 6, nb_epoch = 5, verbose = 1)

### print the keys contained in the history object
#print(history_object.history.keys())

# Visualizing loss
#plt.plot(history_object.history['loss'])
#plt.plot(history_object.history['val_loss'])
#plt.title('model mean squared error loss')
#plt.ylabel('mean squared error loss')
#plt.xlabel('epoch')
#plt.legend(['training set', 'validation set'], loc='upper right')
#plt.ion()
#plt.show()

# Saving model as .h5 format
model.save('model.h5')
print('Model Saved')

