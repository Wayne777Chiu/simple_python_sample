
# refer https://www.kaggle.com/c/titanic
# from keras.activations import deserialize
import numpy as np # linear algebra
import pandas as pd # data processing, csv file I/O

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
    print(decode('|'), '            Titanic - Machine Learing from Disaster  - Practice         ', decode('|'))
    print(decode('|'), '               refer https://www.kaggle.com/c/titanic                   ', decode('|'))
    print(decode('|'), '                              v0.04                                     ', decode('|'))
    print(decode('|'), '            Copyright (c) Wayne Chiu 2021. All Rights Reserved          ', decode('|'))
    print(decode('k--------------------------------------------------------------------------f'))
    print(decode('|'), '                                                                        ', decode('|'))
    print(decode('|'), '   1. Create the sample code for studying Deep Learning                 ', decode('|'))
    print(decode('|'), '              to be continue...                                         ', decode('|'))
    print(decode('p--------------------------------------------------------------------------q'))

def decode(x):
    return (''.join(dicArray.get(i,i.encode('utf-8')).decode('utf-8') for i in x))

license_alarm()
#print(serial_number_plus(),decode('------------------------------------------------------------------------'))
    
import os
import sys
for dirname, _, filenames in os.walk('.\data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
from keras.models import Sequential   #import Sequential model
from keras.layers import Dense        #import Dense loyer


df = pd.read_csv(".\data/train.csv").replace(regex={'female': 1, 'male': 0})   #set female as 1  and male as 0
df = df.fillna(0)     #fill with NaN to zero
dataset = df.values

#set random to noise.
np.random.shuffle(dataset)

# train data set
train_feature_data = dataset[:,np.r_[2,4:8]]  #feature: cloumn_2, column_4, column_5, column_6,, column_7
train_feature_data = np.asarray(train_feature_data).astype('float32')
train_target_data = dataset[:,1] 
train_target_data = np.asarray(train_target_data).astype('float32')

# test data set
test_data_df = pd.read_csv(".\data/test.csv").replace(regex={'female': 1, 'male': 0})  #set female as 1  and male as 0
test_data_df = test_data_df.fillna(0)   #fill with NaN to zero
dataset = test_data_df.values
test_feature_data = dataset[:,np.r_[1,3:7]]   #feature: cloumn_1, column_3, column_4, column_5, column_6
test_feature_data = np.asarray(test_feature_data).astype('float32')

df = pd.read_csv(".\data/gender_submission.csv")
dataset = df.values
test_target_data = dataset[:,1]
test_target_data = np.asarray(test_target_data).astype('float32')

# print(train_feature_data)
# print(train_target_data)

model = Sequential()
model.add(Dense(5, input_shape=(5,), activation="relu"))    #input  layer    5 cell input  
model.add(Dense(6, activation='relu'))                      #hidden layer
model.add(Dense(7, activation='relu'))                      #hidden layer
model.add(Dense(8, activation='relu'))                      #hidden layer
model.add(Dense(1,activation="sigmoid"))                   #output layer 

model.summary()

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# [print(i.shape, i.dtype) for i in model.inputs]
# [print(o.shape, o.dtype) for o in model.outputs]
# [print(l.name, l.input_shape, l.dtype) for l in model.layers]
# print("=====================================================")
model.fit(train_feature_data,train_target_data, epochs=10, batch_size=10, verbose=0)

loss, accuracy = model.evaluate(train_feature_data, train_target_data)
print("Train data accuracy = {:.2f}".format(accuracy))
loss, accuracy = model.evaluate(test_feature_data, test_target_data)
print("Test data accuracy = {:.2f}".format(accuracy))
# Test data prediction
Y_pred = model.predict(test_feature_data)
# print([i for item in Y_pred for i in item])
classes_Y_pred = np.around(Y_pred).astype(int)  
print([i for item in classes_Y_pred for i in item])

output = pd.DataFrame({'PassengerId': test_data_df.PassengerId, 'Survived': [i for item in classes_Y_pred for i in item]})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")

# if __name__ == '__main__':
#     main()