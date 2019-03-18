# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 22:52:49 2019

@author: csrit
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#Getting Started
from sklearn import tree
df=pd.read_csv('iris_df.csv')
df.columns = ['X1', 'X2', 'X3', 'X4', 'Y']
df.head()

#Implementation
from sklearn.model_selection import train_test_split
decision = tree.DecisionTreeClassifier(criterion='gini')
X = df.values[:, 0:4]
Y = df.values[:, 4]
trainX, testX, trainY,testY = train_test_split(X,Y,test_size=0.3)
decision.fit(trainX, trainY)
print('Accuracy: \n', decision.score(testX,testY))

#Visualization
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus as pydot
dot_data = StringIO()
tree.export_graphviz(decision, out_file=dot_data)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_png("tree.png")