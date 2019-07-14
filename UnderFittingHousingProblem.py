#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
x = 2 - 3 * np.random.normal(0, 1, 20)
y = x - 2 * (x ** 2) + 0.5 * (x ** 3) + np.random.normal(-3, 3, 20)
plt.scatter(x,y, s=10)
plt.show()


# In[9]:


from sklearn.linear_model import LinearRegression
np.random.seed(0)
x = 2 - 3 * np.random.normal(0, 1, 20)
y = x - 2 * (x ** 2) + 0.5 * (x ** 3) + np.random.normal(-3, 3, 20)

x = x[:, np.newaxis]
y = y[:, np.newaxis]
model = LinearRegression()
model.fit(x, y)
y_pred = model.predict(x)

plt.scatter(x, y, s=10)
plt.plot(x, y_pred, color='r')
plt.show()


# In[13]:


from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

rmse = (np.sqrt(mean_squared_error(y, y_pred)))
r2 =  r2_score(y, y_pred)
print("RMSE when Linear Regression line is fit : ", rmse)
print("R square when Linear Regression line is fit: ", r2)


# In[ ]:




