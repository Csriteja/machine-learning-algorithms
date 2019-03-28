# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 23:46:27 2019

@author: csrit
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#Getting Started
from sklearn.neighbors import KNeighborsClassifier
df = pd.read_csv('iris_df.csv')
df.columns = ['X1', 'X2','X3', 'X4','Y']
df = df.drop(['X4','X3'],1)
df.head()

#Visualization
sns.set_context("notebook", font_scale=1.1)
sns.set_style("ticks")
sns.lmplot('X1','X2',scatter=True, fit_reg=False, data=df, hue='Y')
plt.ylabel('X2')
plt.xlabel('X1')

#Implementation
from sklearn.model_selection import train_test_split
neighbors = KNeighborsClassifier(n_neighbors=5)
X=df.values[:,0:2]
Y=df.values[:,2]
trainX,testX, trainY, testY = train_test_split(X,Y,test_size=0.3)

neighbors.fit(trainX,trainY)
print('Accuracy: \n', neighbors.score(testX,testY))
pred=neighbors.predict(testX)


