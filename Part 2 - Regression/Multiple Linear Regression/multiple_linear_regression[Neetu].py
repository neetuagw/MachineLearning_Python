# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncod_X = LabelEncoder()
X[:,3] = labelEncod_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the Dummy Variable trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Multiple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting Test set results
y_pred = regressor.predict(X_test)

# Building the Optimal Model using Backward Elimination
import statsmodels.formula.api as stats 
X = np.append(arr = np.ones((len(X),1)).astype(int), values = X, axis = 1)
""""X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = stats.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = stats.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = stats.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 3, 5]]
regressor_OLS = stats.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 3]]
regressor_OLS = stats.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary() """

def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = stats.OLS(y, x).fit()
        maxP = max(regressor_OLS.pvalues).astype(float)
        print(maxP)
        if maxP > sl:
            for j in range(0, numVars-i):
                if (regressor_OLS.pvalues[j].astype(float) == maxP):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x


SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_modeled = backwardElimination(X_opt, SL)