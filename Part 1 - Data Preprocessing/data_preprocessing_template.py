# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 16:05:43 2018

@author: neetu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#Fixing missig values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3]= imputer.transform(X[:,1:3])

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncod_X = LabelEncoder()
X[:,0] = labelEncod_X.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

labelEncod_y = LabelEncoder()
y = labelEncod_y.fit_transform(y)

#Spilliting the dataset into Training and Test Set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
X_standScale = StandardScaler()
X_train = X_standScale.fit_transform(X_train)
X_test = X_standScale.transform(X_test)