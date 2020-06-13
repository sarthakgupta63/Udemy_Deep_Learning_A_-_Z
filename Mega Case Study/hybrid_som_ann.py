#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 20:28:35 2020

@author: sarthakgupta
"""

'''
PART 1 - SOM - UNSUPERVISED LEARNING
'''

# Importing Libraries & Dataset
import numpy as np
import pandas as pd
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
X = sc.fit_transform(X)

# Training SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(X, 100)

# Visualizing SOM
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

# Finding Outliers(Frauds)
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(7,7)], mappings[(6,8)]), axis = 0)
frauds = sc.inverse_transform(frauds)


'''
PART 2 - ANN - SUPERVISED LEARNING
'''

# Creating the matrix of Features
customers = dataset.iloc[:, 1:].values

# Creating Dependant Variable (Target Var)
is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
    if dataset.iloc[i,0] in frauds:
        is_fraud[i] = 1

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler()
customers = sc1.fit_transform(customers)

# Importing Keras libraries & packages
from keras.models import Sequential
from keras.layers import Dense

# Building ANN
classifier = Sequential()
classifier.add(Dense(units = 2, kernel_initializer='uniform', activation='relu', input_dim=15))
classifier.add(Dense(units = 1, kernel_initializer='uniform', activation='sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

summ = classifier.summary()

classifier.fit(customers, is_fraud, batch_size=1, epochs=2)

# Predicting the probabilities of Fraud
y_pred = classifier.predict(customers)
y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis = 1)

# Sorting the array according to probabality of fraud
y_pred = y_pred[y_pred[:,1].argsort()]


