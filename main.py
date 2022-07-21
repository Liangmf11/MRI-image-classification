# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 02:20:06 2022

@author: kjk
"""
import cv2
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense



image_directory = 'C:/Users/kjk/.spyder-py3/FP/datasets/'

no_tumor_images = os.listdir(image_directory + 'no/')   
yes_tumor_images = os.listdir(image_directory + 'yes/')   
dataset = []
label = []

INPUT_SIZE = 64
#print(no_tumor_images)

for i, image_name in enumerate(no_tumor_images):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_directory + 'no/' + image_name)
        image=Image.fromarray(image, 'RGB')
        image=image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(0)
        
for i, image_name in enumerate(yes_tumor_images):
    if(image_name.split('.')[1]=='jpg'):
        image = cv2.imread(image_directory + 'yes/' + image_name)
        image=Image.fromarray(image, 'RGB')
        image=image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)

        
#print(dataset)
#print(len(label))

dataset=np.array(dataset)
label = np.array(label)

x_train, x_test, y_train, y_test = train_test_split(dataset,label,test_size = 0.2, random_state=0)

#print(x_test.shape)


x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)

#Model Building
#64,64,3

model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(INPUT_SIZE,INPUT_SIZE,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model = Sequential()
model.add(Conv2D(32, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model = Sequential()
model.add(Conv2D(64, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))


model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

history=model.fit(x_train,y_train,batch_size=16, verbose=1, epochs=10, validation_data=(x_test,y_test),shuffle=False)
model.summary()
tf.keras.utils.plot_model(model,
                          to_file="model.png",
                          show_shapes=True,
                          expand_nested=True)
'''
y_pred = model.predict(x_test)
y_final=np.argmax(y_pred,axis=1)
#print(y_final)

from sklearn.metrics import confusion_matrix

#Generate the confusion matrix
cf_matrix = confusion_matrix(y_test, y_final)

import seaborn as sns

ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')

ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['True','False'])
ax.yaxis.set_ticklabels(['Positive','negative'])

## Display the visualization of the Confusion Matrix.
plt.show()


loss_train = history.history['accuracy']
loss_val = history.history['val_accuracy']
epoch_range = range(10)
plt.plot(epoch_range, loss_train, 'g', label='Training accuracy')
plt.plot(epoch_range, loss_val, 'b', label='validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

loss_train = history.history['loss']
loss_val = history.history['val_loss']
epoch_range = range(10)
plt.plot(epoch_range, loss_train, 'g', label='Training loss')
plt.plot(epoch_range, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
'''
model.save('BrainTumor10epochs.h5')
