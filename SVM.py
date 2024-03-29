# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 23:50:29 2019

@author: csrit
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#Getting Started
from sklearn import svm
df = pd.read_csv('iris_df.csv')
df.columns = ['X4', 'X3', 'X1', 'X2', 'Y']
df = df.drop(['X4', 'X3'], 1)
df.head()

#Implementation
from sklearn.model_selection import train_test_split
support = svm.SVC()
X = df.values[:, 0:2]
Y = df.values[:, 2]
trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3)
support.fit(trainX, trainY)
print('Accuracy: \n', support.score(testX, testY))
pred = support.predict(testX)

#Visualization
sns.set_context("notebook", font_scale=1.1)
sns.set_style("ticks")
sns.lmplot('X1','X2',scatter=True, fit_reg=False, data=df, hue='Y')
plt.ylabel('X2')
plt.xlabel('X1')