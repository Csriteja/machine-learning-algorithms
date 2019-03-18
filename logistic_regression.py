# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 22:43:17 2019

@author: csrit
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#Getting Started
from sklearn.linear_model import LogisticRegression
df=pd.read_csv('logistic_regression_df.csv')
df.columns = ['X','Y']
df.head()

#Implementation

logistic=LogisticRegression()
X=(np.asarray(df.X)).reshape(-1,1)
Y=(np.asarray(df.Y)).ravel()
logistic.fit(X,Y)
logistic.score(X,Y)
print('Coefficient: \n',logistic.coef_)
print('Intercept: \n',logistic.intercept_)
print('R^2 value: \n',logistic.score(X,Y))

#Visualization

sns.set_context("notebook",font_scale=1.1)
sns.set_style("ticks")
sns.regplot('X','Y',data=df,logistic=True)
plt.ylabel('Probability')
plt.xlabel('Explanatory')