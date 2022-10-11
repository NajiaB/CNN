# -*- coding: utf-8 -*-
"""
Created on Mon May 23 20:59:36 2022

@author: Najia
"""

from tensorflow import keras
from keras.datasets import mnist
from keras.utils import to_categorical

### first step : Loading data + put it into the right shape 

(X_train, y_train), (X_test, y_test) = mnist.load_data() 

num_train  = X_train.shape[0]  #60000
num_test   = X_test.shape[0]      #10000 image

img_height = X_train.shape[1]   #28
img_width  = X_train.shape[2]    #28
X_train = X_train.reshape((num_train, img_height, img_width))
X_train=X_train.astype('float32')/255

X_test  = X_test.reshape((num_test, img_height, img_width))
X_test  = X_test.astype('float32')/255

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

### second step : building a model : Conv-Relu-MaxPool layers

from keras import layers
from keras import models

model=models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))

model.summary()

### third step : flatten the 3D output into 1D output 
#bcz the next layer is DENSE and only takes 1D tensors as input

model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))

model.summary()

###Third step : train this convnet model on MNIST dataset

model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,epochs=5,batch_size=64)

### fourth step : testing

test_loss, test_acc = model.evaluate(X_test, y_test)
test_acc
