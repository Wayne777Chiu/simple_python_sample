
# refer https://www.kaggle.com/c/titanic
# from keras.activations import deserialize
from typing import Optional
import numpy as np # linear algebra
import pandas as pd
from scipy.sparse import data
import matplotlib.pyplot as plt

serial_number  = 0

def serial_number_plus():
    global serial_number
    serial_number += 1
    return serial_number


dicArray = {
    'h': b'\xe2\x94\x9c',
    'l': b'\xE2\x94\x82',
    'm': b'\xE2\x94\x80',
    'e': b'\xE2\x94\x94',
    '+': b'\xE2\x95\x94',
    '%': b'\xE2\x95\x97',
    'p': b'\xE2\x95\x9A',
    'q': b'\xE2\x95\x9D',
    '-': b'\xE2\x95\x90',
    '|': b'\xE2\x95\x91',
    'k': b'\xE2\x95\xA0',
    'f': b'\xE2\x95\xA3',
}

def license_alarm():
    print(decode('+--------------------------------------------------------------------------%'))
    print(decode('|'), '        Digit Recognizer-  with the famous MNIST data  - Practice       ', decode('|'))
    print(decode('|'), '               refer https://www.kaggle.com/c/digit-recognizer/code     ', decode('|'))
    print(decode('|'), '                              v0.01                                     ', decode('|'))
    print(decode('|'), '            Copyright (c) Wayne Chiu 2021. All Rights Reserved          ', decode('|'))
    print(decode('k--------------------------------------------------------------------------f'))
    print(decode('|'), '                                                                        ', decode('|'))
    print(decode('|'), '   1. Create the sample code for studying Deep Learning                 ', decode('|'))
    print(decode('|'), '              to be continue...                                         ', decode('|'))
    print(decode('p--------------------------------------------------------------------------q'))

def decode(x):
    return (''.join(dicArray.get(i,i.encode('utf-8')).decode('utf-8') for i in x))

def showplt(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)  # Set the vertical range to [0-1]
    plt.show()

license_alarm()
#print(serial_number_plus(),decode('------------------------------------------------------------------------'))
    
import os
import sys
for dirname, _, filenames in os.walk('./kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title("Train History")
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.show()



# from keras.models import Sequential           #import Sequential model
# from keras.layers import Dense,Dropout        #import Dense loyer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.layers import Conv2D, MaxPool2D
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import RMSprop
from tensorflow.python.keras.layers.core import Dropout # data processing, csv file I/O
# from tensorflow.keras.utils import np_utils

train_df = pd.read_csv("./kaggle/input/digit-recognizer/train.csv")
test_df = pd.read_csv("./kaggle/input/digit-recognizer/test.csv")

train_df.head
test_df.head


#==========2 filter train data x and y
# 
X_train_data = train_df.drop('label',axis=1)/255
y_train_data= pd.get_dummies(train_df.label)

X_train, X_valid, y_train, y_valid = train_test_split(X_train_data, y_train_data, train_size=0.8, test_size=0.2, random_state=50)

X_train = X_train.values.reshape(-1, 28, 28, 1)
X_valid = X_valid.values.reshape(-1, 28, 28, 1)

EPOCH = 30
LEARNING_RATE = 0.00025

model = Sequential()

model.add(Conv2D(32, (5,5), activation="relu", padding="same", data_format="channels_last", input_shape=(28,28,1)))
model.add(Conv2D(32, (5,5), activation="relu", padding="same", data_format="channels_last"))
model.add(MaxPool2D(pool_size=(2,2), data_format="channels_last"))
model.add(Dropout(0.25))

model.add(Conv2D(64, (5,5), activation="relu", padding="same", data_format="channels_last"))
model.add(Conv2D(64, (5,5), activation="relu", padding="same", data_format="channels_last"))
model.add(MaxPool2D(pool_size=(2,2), data_format="channels_last"))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))

Decay = 5* LEARNING_RATE / EPOCH
Optimizer= RMSprop(learning_rate=LEARNING_RATE, rho=0.9, epsilon=1e-08, decay=Decay)
model.compile(optimizer=Optimizer, loss="categorical_crossentropy", metrics=['accuracy'])

model.summary()

lr_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
train_history = model.fit(X_train, y_train, epochs=EPOCH, validation_data=(X_valid,y_valid), verbose=2, callbacks=[lr_reduction])

show_train_history(train_history, 'accuracy', 'val_accuracy')


show_train_history(train_history, 'loss', 'val_loss')


# print(X_train)
# y_train.head
