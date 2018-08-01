# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 22:22:56 2018

@author: neetu
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#Fitting Linear Regression Model to the dataset
from sklearn.linear_model import LinearRegression
lin_regressor = LinearRegression()
lin_regressor.fit(X, y)

#Fitting Polynomial Linear Regression Model to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_regressor = PolynomialFeatures(degree=4)
X_poly = poly_regressor.fit_transform(X)

lin_regressor2 = LinearRegression()
lin_regressor2.fit(X_poly, y)

#Visuaising Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_regressor.predict(X), color='blue')
plt.title('Level vs Salary Linear Regression Model')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

#Visuaising Polynomial Regression results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_regressor2.predict(poly_regressor.fit_transform(X_grid)), color='blue')
plt.title('Level vs Salary Polynomial Regression Model')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

#Predicting a new result with Linear Regressor
lin_regressor.predict(6.5)

#Predicting a new result with Polynomial Regressor
lin_regressor2.predict(poly_regressor.fit_transform(6.5))