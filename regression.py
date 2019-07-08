#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 23:13:40 2019

@author: neetika
"""

from sklearn import linear_model
import numpy as np	

F, N = [int(x) for x in input().split()]
X = []
y = []
test_data = []

for j in range(N):
    x_y = [float(x) for x in input().split()]
    X.append(x_y[:-1])
    y.append(x_y[-1])
    
X = np.array(X)
y = np.array(y)

no_of_test = int(input())
for i in range(no_of_test):
    test_data.append([float(x) for x in input().split()])
   
lm = linear_model.LinearRegression()
lm.fit(X, y)

predictions = lm.predict(test_data)

for i in range(no_of_test):
    print(round(predictions[i], 2))
