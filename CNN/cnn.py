#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 19:55:25 2020

@author: sarthakgupta
"""

"""
IMAGE CLASSIFICATION - CATS VS DOGS
"""

'''
BUILDING CNN
'''

# Importing Keras libraries & packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Building CNN
classifier = Sequential()                                                                       # INITIALIZATION
                                                                                                #       |
classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'))             #  CONVOLUTION
                                                                                                #       |
classifier.add(MaxPooling2D(pool_size = (2,2)))                                                 #  MAX POOLING
                                                                                                #       |
classifier.add(Convolution2D(32, 3, 3, activation='relu'))                                      #  CONVOLUTION
                                                                                                #       |
classifier.add(MaxPooling2D(pool_size = (2,2)))                                                 #  MAX POOLING
                                                                                                #       |
classifier.add(Convolution2D(64, 3, 3, activation='relu'))                                      #  CONVOLUTION
                                                                                                #       |
classifier.add(MaxPooling2D(pool_size = (3,3)))                                                 #  MAX POOLING
                                                                                                #       |
classifier.add(Flatten())                                                                       #  FLATTENING
                                                                                                #       |
classifier.add(Dense(units = 128, activation = 'relu'))                                         # FULLY CONNECTED
                                                                                                #       |
classifier.add(Dense(units = 1, activation = 'sigmoid'))                                        #     OUTPUT

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

summ = classifier.summary()


'''
IMAGE DATA PREPROCESSING & AUGMENTATION
'''
# Import package for image generation & augmentation ready for the model
from keras.preprocessing.image import ImageDataGenerator

# Initializing objects for augmentation of data
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

# generating train and test data separeately using the above defined datagenerator objects
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (128, 128),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (128, 128),
                                            batch_size = 32,
                                            class_mode = 'binary')

# Fitting, Genrating, training, evaluating, etc. All at once
classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000)
















