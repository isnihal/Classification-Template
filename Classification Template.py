# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 12:47:03 2017

@author: nihal
"""

#Importing Libraries
import numpy as np
import pandas as pd

#Importing dataset
dataset=pd.read_csv("Social_Network_Ads.csv")
x=dataset.iloc[:,2:4].values
y=dataset.iloc[:,-1].values

#Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x=scaler.fit_transform(x)

#Splitting sets
from sklearn.cross_validation import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=0.8)

#Logistic classification
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(xtrain,ytrain)