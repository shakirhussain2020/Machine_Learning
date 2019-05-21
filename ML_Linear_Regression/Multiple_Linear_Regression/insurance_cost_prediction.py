# -*- coding: utf-8 -*-
"""
Created on Mon May 20 23:58:55 2019

@author: user
"""
import pandas as pd
import numpy as np

#Importing the dataset
dataset=pd.read_csv('insurance.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,6].values

#Encoding categorical variables
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X  = LabelEncoder()
X[:,1] = labelencoder_X.fit_transform(X[:,1])
X[:,4] = labelencoder_X.fit_transform(X[:,4])
X[:,5] = labelencoder_X.fit_transform(X[:,5])
onehotencoder = OneHotEncoder(categorical_features=[5])
X=onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X=X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

#Building optimal model using backward elimination technique
import statsmodels.formula.api as sm
X=np.append(arr=np.ones((1338,1)).astype(int),values= X, axis=1)
X_opt = X[:,[0,1,2,3,4,5,6,7,8]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,1,2,3,4,6,7,8]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,2,3,4,6,7,8]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_opt, y, test_size = 0.2, random_state = 0)
# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred_opt = regressor.predict(X_test)