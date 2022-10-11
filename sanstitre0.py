# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 18:21:24 2022

@author: Najia
"""

from keras.datasets import boston_housing
(train_data,train_target),(test_data,test_target)=boston_housing.load_data()

##normalisation
mean=train_data.mean(axis=0)
train_data-=mean

std=train_data.std(axis=0)
train_data/=std

test_data-=mean
test_data /=std

##architecture du réseau
!pip install seaborn
from keras import layers
from keras import models

def build_model():
    model=models.Sequential()
    model.add(layers.Dense(64,activation='relu',input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
    return(model)
