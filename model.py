#!/usr/bin/env python3

# importing functions

import pandas as pd
import numpy as np
import cv2
import sklearn
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from random import randint
import csv
from scipy import ndimage
from sklearn.utils import shuffle

# importing data

samples = []
with open('data/driving_log.csv') as csvfile:
    reader =csv.reader(csvfile)
    for line in reader:
        samples.append(line)


# creating generator function

def generator (samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]  
            images = []
            measurements = []

            for batch_sample in batch_samples:
                filename = 'data/IMG/' + batch_sample[0].split('/')[-1]
                image = ndimage.imread(filename)
                images.append(image)
                measurement = float(batch_sample[3])
                measurements.append(measurement)

            # creating vertical mirror information
            augmented_images , augmented_measurements = [],[]

            for image,measurements in zip (images,measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurements)
                augmented_images.append(cv2.flip(image,-1))
                augmented_measurements.append(measurement *-1)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)

            yield sklearn.utils.shuffle(X_train, y_train)
            
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

batch_size=32

# training the data

train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

from keras.models import Sequential,Model
from keras.layers import Convolution2D,Flatten,Dense,Lambda,Cropping2D,Dropout

# creating the convulotional neural network

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))

# trim image to only see section with road
model.add(Cropping2D(cropping=((70,25),(0,0))))

#layer 1- Convolution, no of filters- 24, filter size= 5x5, stride= 2x2
model.add(Convolution2D(24,5,5,subsample=(2,2),activation = "relu"))

#layer 2- Convolution, no of filters- 36, filter size= 5x5, stride= 2x2
model.add(Convolution2D(36,5,5,subsample=(2,2),activation = "relu"))

#layer 3- Convolution, no of filters- 48, filter size= 5x5, stride= 2x2
model.add(Convolution2D(48,5,5,subsample=(2,2),activation = "relu"))

#layer 4- Convolution, no of filters- 64, filter size= 3x3, stride= 1x1
model.add(Convolution2D(64,3,3,activation = "relu"))

#layer 5- Convolution, no of filters- 64, filter size= 3x3, stride= 1x1
model.add(Convolution2D(64,3,3,activation = "relu"))

#flatten image from 2D to side by side
model.add(Flatten())

#layer 6- fully connected layer 1
model.add(Dense(100))

#Adding a dropout layer to avoid overfitting. Here we are have given the dropout rate as 30% after first fully connected layer
model.add(Dropout(0.30))

#layer 7- fully connected layer 1
model.add(Dense(50))

#layer 8- fully connected layer 1
model.add(Dense(10))

#layer 9- fully connected layer 1
model.add(Dense(1))
#here the final layer will contain one value as this is a regression problem and not classification

model.compile(loss='mse',optimizer = 'adam')
model.fit_generator(train_generator,steps_per_epoch=np.ceil(len(train_samples)/batch_size),validation_data=validation_generator,validation_steps=np.ceil(len(validation_samples)/batch_size),epochs=5, verbose=1)

model.save('model.h5')

print ("Done")


#history_object = model.fit_generator(train_generator, samples_per_epoch =
    #len(train_samples), validation_data = 
    #validation_generator,
    #nb_val_samples = len(validation_samples), 
    #nb_epoch=5, verbose=1)

### print the keys contained in the history object
#print(history_object.history.keys())

### plot the training and validation loss for each epoch
#plt.plot(history_object.history['loss'])
#plt.plot(history_object.history['val_loss'])
#plt.title('model mean squared error loss')
#plt.ylabel('mean squared error loss')
#plt.xlabel('epoch')
#plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('images/model_mean_squared_error_loss.jpg')
plt.show()