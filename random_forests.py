# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 23:45:07 2019

@author: csrit
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#Getting Started
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv('iris_df.csv')
df.columns = ['X1', 'X2', 'X3', 'X4', 'Y']
df.head()

