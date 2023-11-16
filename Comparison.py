#!/usr/bin/env python
# coding: utf-8

# https://github.com/sahilichake/Indian-Crime-Data-Analysis-Forecasting/blob/main/Indian-Crime-Data-Analysis-Forecasting.ipynb

# Implement machine learning model

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


# In[2]:


data = pd.read_csv("crime.csv")


# In[3]:


data.columns


# In[4]:


min_val = data['TOTAL IPC CRIMES'].min()
max_val = data['TOTAL IPC CRIMES'].max()
range_val = (max_val - min_val) / 4
low = min_val + range_val
medium = low + range_val
high = medium + range_val


# In[5]:


def get_crime_level(crime_count):
    if crime_count <= low:
        return 1
    elif crime_count <= medium:
        return 2
    elif crime_count <= high:
        return 3
    else:
        return 4

data['CRIME_LEVEL'] = data['TOTAL IPC CRIMES'].apply(get_crime_level)


# In[6]:


crime_level_count = data['CRIME_LEVEL'].value_counts()
crime_level_count


# In[7]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# fit and transform the STATE/UT column using the LabelEncoder
data["STATE/UT_encoded"] = le.fit_transform(data["STATE/UT"])

# fit and transform the DISTRICT column using the LabelEncoder
data["DISTRICT_encoded"] = le.fit_transform(data["DISTRICT"])


# In[8]:


grouped_state = data[["STATE/UT", "STATE/UT_encoded"]].groupby("STATE/UT").first()
grouped_state


# In[9]:


grouped_district = data[["DISTRICT", "DISTRICT_encoded"]].groupby("DISTRICT").first()
grouped_district


# In[10]:


data.columns


# Linear Regression

# In[11]:


X = data.drop(['CRIME_LEVEL','STATE/UT', 'DISTRICT','TOTAL IPC CRIMES'], axis=1)
y = data['TOTAL IPC CRIMES']
X.columns


# In[12]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[13]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[14]:


from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train,y_train)
lr_pred = lr.predict(X_test)


# In[15]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix

# Evaluate the regression model
lr_score = lr.score(X_test, y_test)
print('Linear Regression score : ',lr_score)
print()
print('Mean_absolute_error  = ',mean_absolute_error(lr_pred,y_test))
print()
print('Mean_squared_error   = ',mean_squared_error(lr_pred,y_test))
print()
print('R2_score             = ',r2_score(lr_pred,y_test))
print()
R2 = r2_score(lr_pred,y_test)
adj_R2 = 1-((1-R2)*(len(y)-1)/(len(y)-X.shape[1]-1))
print('Adjusted_R2         = ',adj_R2)


# In[16]:


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log = LogisticRegression(max_iter=1000)
log.fit(X_train_scaled, y_train)
log_pred = log.predict(X_test_scaled)


# In[17]:


log_score = log.score(X_test, y_test)
log_prob = log.predict_proba(X_test)[:, 1]
print('Logistic regression score : ',log_score)
print()
print("Confusion matrix:\n")
print(confusion_matrix(y_test, log_pred))
print()

