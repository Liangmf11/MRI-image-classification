# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 05:14:22 2022

@author: kjk
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import os
import cv2


image_directory = 'C:/Users/kjk/.spyder-py3/FP/datasets/'

no_tumor_images = os.listdir(image_directory + 'no/')   
yes_tumor_images = os.listdir(image_directory + 'yes/')  

dataset = []
label = []

#print(no_tumor_images)

for i, image_name in enumerate(no_tumor_images):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_directory + 'no/' + image_name)
        image = cv2.resize(image, (200,200))
        dataset.append(np.array(image))
        label.append(0)
        
for i, image_name in enumerate(yes_tumor_images):
    if(image_name.split('.')[1]=='jpg'):
        image = cv2.imread(image_directory + 'yes/' + image_name)
        image = cv2.resize(image, (200,200))
        dataset.append(np.array(image))
        label.append(1)
        
dataset = np.array(dataset)
label = np.array(label)

X_updated = dataset.reshape(len(dataset), -1)

xtrain, xtest, ytrain, ytest = train_test_split(X_updated, label, random_state=10,
                                               test_size=.20)

xtrain = xtrain/255
xtest = xtest/255

from sklearn.decomposition import PCA
pca = PCA(.98)
# pca_train = pca.fit_transform(xtrain)
# pca_test = pca.transform(xtest)
pca_train = xtrain
pca_test = xtest

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import warnings
warnings.filterwarnings('ignore')

lg = LogisticRegression(C=0.1)
lg.fit(pca_train, ytrain)
sv = SVC()
sv.fit(pca_train, ytrain)

print("Training Score:", lg.score(pca_train, ytrain))
print("Testing Score:", lg.score(pca_test, ytest))