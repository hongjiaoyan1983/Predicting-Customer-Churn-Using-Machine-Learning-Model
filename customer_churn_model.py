#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 15:59:39 2017

@author: sonishivama@gmail.com
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# fix random seed for reproducibility
np.random.seed(7)

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
 #Not ordinal, therefore we need to creat dummy variable for countries (as more than 2) --> includes also a new measure
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X=X[:, 1:]#to avoid dummy variable trap, we need to drop one of countries


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#fitting xboost to the training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#applying k-fold cross validation
from sklearn.cross_validation import cross_val_score
accuracies = cross_val_score(estimator=classifier , X=X_train, y=y_train, cv=10)
accuracies.mean()
accuracies.std()

#apply grid search for parameter tuning
from sklearn.model_selection import GridSearchCV
parameters = [{'learning_rate' : [0.0,0.1,0.2,0.3], 'gamma' : [0.01,0.1,0.25,0.50] , 'base_score' : [0.1,0.2,0.5,0.75]}]
grid_search = GridSearchCV(estimator= classifier,
                           param_grid = parameters,
                           scoring= 'accuracy',
                           cv = 4)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_params = grid_search.best_params_
