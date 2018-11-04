# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 23:50:44 2018

@author: ambujesh
"""

import pandas
import numpy as np
import scipy
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

le = preprocessing.LabelEncoder()

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

dataset['class'] = le.fit_transform(dataset['class'])

#dataset = pandas.concat([dataset,pandas.get_dummies(dataset['class'], prefix='class')],axis=1)

dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

dataset.hist()
plt.show()

scatter_matrix(dataset)
plt.show()

X = dataset.iloc[:, :4].values
Y = dataset.iloc[:, 4].values

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20, random_state = 0)

# Preparation of model

Classifier_1 = DecisionTreeClassifier(criterion="entropy")
Classifier_2 = KNeighborsClassifier()
Classifier_3 = LogisticRegression()



Classifier_1.fit(X_train, Y_train)
Y_pred_1 = Classifier_1.predict(X_test)
print(accuracy_score(Y_test, Y_pred_1))

Classifier_2.fit(X_train, Y_train)
Y_pred_2 = Classifier_2.predict(X_test)
print(accuracy_score(Y_test, Y_pred_2))

Classifier_3.fit(X_train, Y_train)
Y_pred_3 = Classifier_3.predict(X_test)
print(accuracy_score(Y_test, Y_pred_3))
