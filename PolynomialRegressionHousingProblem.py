#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

import operator

#demonstrating Polynomial Regression for random data first.
np.random.seed(0)
x = 2- 3 * np.random.normal(0, 1, 20)
y = x - 2 * (x**2) + 0.5 * (x ** 3)+ np.random.normal(-3, 3, 20)

x = x[:, np.newaxis]
y = y[:, np.newaxis]

#Polynomial Transformation
polynomial_features = PolynomialFeatures(degree=2)
x_poly = polynomial_features.fit_transform(x)

model = LinearRegression()
model.fit(x_poly, y)
y_poly_pred = model.predict(x_poly)

rmse = (np.sqrt(mean_squared_error(y, y_poly_pred)))
r2 = r2_score(y, y_poly_pred)
print("Root mean squared Error: ", rmse)
print("R Square: ", r2)

plt.scatter(x, y, s=10)

sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(x, y_poly_pred), key = sort_axis)
x, y_poly_pred = zip(*sorted_zip)
plt.plot(x, y_poly_pred, color = 'm')
plt.show()


# In[9]:


#For Housing Boston Dataset, using Polynomial Regression to minimize error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pandas as pd

boston_dataset = load_boston()

boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)

boston['MEDV'] = boston_dataset.target
print(boston.head())

X = pd.DataFrame(np.c_[boston['LSTAT'], boston['RM']], columns = ['LSTAT','RM'])
Y = boston['MEDV']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

def create_polynomial_regression_model(degree):
    polynomial_features = PolynomialFeatures(degree)
    X_train_poly = polynomial_features.fit_transform(X_train)
    X_test_poly = polynomial_features.fit_transform(X_test)
    
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, Y_train)
    
    #predicting y on training data
    y_train_predicted = poly_model.predict(X_train_poly)
    
    #predicting y on test data
    y_test_predicted = poly_model.predict(X_test_poly)
    
    
    #evaluation on training data
    rmse_train = (np.sqrt(mean_squared_error(Y_train, y_train_predicted)))
    r2_train = r2_score(Y_train, y_train_predicted)
    
    #evaluation on test data
    rmse_test = (np.sqrt(mean_squared_error(Y_test, y_test_predicted)))
    r2_test = r2_score(Y_test, y_test_predicted)
    
    print("Model Performance with Polynomial regression fit to data is as follows: ")
    print("Root Mean Squared Error for Training Data: ", rmse_train)
    print("R Square value for Training Data: ", r2_train)
    print("Root Mean Squared Error for Test Data: ", rmse_test)
    print("R Square value for Test Data: ", r2_test)


# In[10]:


create_polynomial_regression_model(2)


# In[ ]:




