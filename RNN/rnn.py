#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 15:03:29 2020

@author: sarthakgupta
"""

'''
Recurrent Neural Networks-
Implementing a LSTM model on Google Stock Prices
'''

'''
DATA PREPROCESSING
'''

# Importing Libraraies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
training_dataset = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = training_dataset.iloc[:, 1:2].values

# Normalizing the data
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)

# Creating dataset with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train = np.array(X_train)
y_train = np.array(y_train)

# Reshaping according to Keras input shape
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


'''
BUILDING THE RNN
'''

# Import required libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialize the RNN
regressor = Sequential()

# Adding the first LSTM Layer with Dropout
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(rate = 0.2))

# Adding Consecutive LSTM Layers
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(rate = 0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(rate = 0.2))


# Adding the final layer
regressor.add(LSTM(units = 50))
regressor.add(Dropout(rate = 0.2))

# Adding the Output Layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Summary of the model
summ = regressor.summary()

# Fitting the model to the training set
regressor.fit(X_train, y_train, batch_size = 32, epochs = 100)


'''
MAKING THE PREDICTIONS & VISUALIZING THE RESULTS
'''

# Getting Real Stock Prices for 2017
test_dataset = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = test_dataset.iloc[:, 1:2].values

# Getting Predicted Stock Price Values for 2017
# Training + Test set
dataset_total = pd.concat((training_dataset['Open'], test_dataset['Open']), axis = 0)

# Get last 60 of training and rest from test sets
inputs = dataset_total[len(dataset_total) - len(test_dataset) - 60 : ].values

# reshape
inputs = inputs.reshape(-1,1)

# Scaling - only transform using 'sc'
inputs = sc.transform(inputs)

# Reshape to get 3D structure
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Predict using the model - 'regressor'
pred_stock_price = regressor.predict(X_test)

# Getting values back from scaled values
pred_stock_price = sc.inverse_transform(pred_stock_price)


# Visualizing the Results
plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
plt.plot(pred_stock_price, color='blue', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

