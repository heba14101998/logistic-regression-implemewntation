# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 16:50:58 2022

@author: Heba Mohamed
"""

from sklearn.model_selection import train_test_split
from sklearn import datasets
from logistic_regression import LogisticRegression
import numpy as np

def accuracy(y_actual, y_pred):
    return np.sum(y_actual == y_pred) / len(y_actual)


bc_data = datasets.load_breast_cancer()
X, y = bc_data.data, bc_data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 2022)

reg_model = LogisticRegression(lr=0.001, n_iters=1000)
reg_model.fit(X_train, y_train)
y_pred = reg_model.predict(X_test)

print("LR classification accuracy:",accuracy(y_test, y_pred),'\n\n')