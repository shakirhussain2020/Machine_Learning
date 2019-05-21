# -*- coding: utf-8 -*-
"""
Created on Mon May 20 09:33:20 2019

@author: User
"""
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('Kaggle_dataset.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=42)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)

plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.xlabel("SAT")
plt.ylabel("GPA")
plt.title("SAT Vs GPA Training Set")
plt.show()

plt.scatter(X_test,y_test,color='green')
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.xlabel("SAT")
plt.ylabel("GPA")
plt.title("SAT Vs GPA Testing Set")
plt.show()
