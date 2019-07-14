#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns


# In[10]:


from sklearn.datasets import load_boston
boston_dataset = load_boston()
ks = boston_dataset.keys()


# In[12]:


print(ks)


# In[13]:


print(ks)


# In[6]:


ks = boston_dataset.keys()
print(ks)


# In[7]:


ks = boston_dataset.keys()
print(boston_dataset['data'])


# In[8]:


print(boston_dataset['data'][0])


# In[9]:


print(ks)


# In[14]:


print(boston_dataset['target'])


# In[15]:


print(boston_dataset['target'][0])


# In[16]:


print(boston_dataset['data'][0][12])


# In[17]:


print(len(boston_dataset['data'][0])


# In[18]:


print(len(boston_dataset['data'][0]))


# In[19]:


print(boston_dataset['data'][0][12])


# In[20]:


print(boston_dataset['feature_names'])


# In[21]:


print(boston_dataset['filename'])


# In[22]:


boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
print(boston)


# In[23]:


print(boston.head)


# In[24]:


print(boston.head())


# In[25]:


boston['MEDV'] = boston_dataset.target
print(boston.head())


# In[26]:


boston.isnull.sum()


# In[27]:


print(boston.isnull.sum())


# In[28]:


boston.isnull().sum()


# In[29]:


sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(boston['MEDV'], bins=30)
plt.show()


# In[30]:


plt.figure(figsize=(20, 5))
features = ['LSTAT', 'RM']
target = boston['MEDV']

for i, col in enumerate(features):
    print(i)


# In[31]:


for i, col in enumerate(features):
    print(col)


# In[32]:


print(len(features))


# In[33]:


for i, col in enumerate(features):
    plt.subplot(1, len(features), i+1)
    x = boston[col]
    y = target
    plt.scatter(x, y, marker='o')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('MEDV')


# In[34]:


X = pd.DataFrame(np.c_[boston['LSTAT'], boston['RM']], columns = ['LSTAT','RM'])
Y = boston['MEDV']


# In[35]:


print(X)


# In[36]:


print(Y)


# In[37]:


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[38]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)


# In[44]:


from sklearn.metrics import r2_score

y_train_predict = lin_model.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
r2 = r2_score(Y_train, y_train_predict)

print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

y_test_predict = lin_model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
r2 = r2_score(Y_test, y_test_predict)
                    
print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))


# In[45]:


print(boston_dataset['data'][0])


# In[ ]:




