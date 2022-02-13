# Data Preprocessing Tools

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Background.csv')
X = dataset.iloc[:, :-1].values
#print(X)
y = dataset.iloc[:, -1].values # NOTICE! .iloc[all the rows, only the last column]
#print(y)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1) Data preprocessing

# Encoding the Independent Variable (categorical data)

# What is happening here? = Turning the string columns (countries) into unique binary vectors

# 'One hot encoding' = Splitting a column up using the unique values. Creating binary vectors for each unique value
# 'ct' (object of the 'ColumnTransformer' class) = Creating an instance of the 'ColumnTransformer' class
# 'ColumnTransformer(transformers=[(The kind of transformation, What kind encoding, index of the columns we want to encode)], remainder = 'passthrough')'


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1) Data preprocessing

# Splitting the dataset into the Training set and Test set

# Note to self! = Split the data before feature scaling!
# Test set = future data
# Feature scaling = scaling the features so that they all take values in the same scale
# 80/20 split
# 'test_size' = 20% for the test set

# 'X_train' The matrix of the features of the training set
# 'X_test' The matrix of the features of the test set
# 'y_train' The dependent variable of the training set
# 'y_test' The dependent variable of the test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
#print(X_train) # The matrix of the features of the training set
#print(X_test) # The matrix of the features of the test set
#print(y_train) # The dependent variable of the training set
#print(y_test) # The dependent variable of the test set

# REMINDER MULTIPLE LINEAR REGRESSION DOES NOT NEED FEATURE SCALING!!!

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2) Multiple Linear Regression.

# Training the Multiple Linear Regression model on the Training set
# MUlTIPLE LINEAR REGRESSION does not involve dummy variables
# The class will automatically select the most relevant model to use

from sklearn.linear_model import LinearRegression
regressor = LinearRegression() # An instance of the 'LinearRegression' class
regressor.fit(X_train, y_train) # Matrix of features, Independent variable.

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2) Multiple Linear Regression.

# Predicting the Test set results

# '.set_printoptions(precision=2)' = Method for displaying values to a certain decimal point
# The 'len' function captures the length of the vector.
# 'np.concatenate((Vector of predicted profits (Display vertically), vector of real profits), axis (0 = means vertical, 1 = horizontal))'
# '.reshape(length of vector, number of columns)'

y_pred = regressor.predict(X_test) # Vector of dependent variables.
np.set_printoptions(precision = 2) # Displaying numerical values to 2 D.P
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)) # Concatenate two vectors vertically (reshaping vectors)
