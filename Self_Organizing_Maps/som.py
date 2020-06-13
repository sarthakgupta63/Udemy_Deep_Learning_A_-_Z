#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 12:14:45 2020

@author: sarthakgupta
"""

# Importing Libraries
import numpy as np
import pandas as pd

# Importing Dataset and splitting into X and y(not the target variable)
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[: , -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
X = sc.fit_transform(X)

## Training the SOM
# Importing MiniSom
from minisom import MiniSom

# Creating SOM object
som = MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.5)

# Initializing the weigths
som.random_weights_init(X)

# Training the SOM
som.train_random(data = X, num_iteration = 100)

## Visualization (from scratch) - Tricky Part
from pylab import bone, plot, colorbar, show, pcolor
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

# Finding the Outliers(Frauds) using the inverse mapping
mappings = som.win_map(X)
# frauds = mappings[(8,6)]
frauds = np.array(mappings[(2,3)])
frauds = sc.inverse_transform(frauds)
