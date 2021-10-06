
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
    print(decode('|'), '                              v0.02                                     ', decode('|'))
    print(decode('|'), '            Copyright (c) Wayne Chiu 2021. All Rights Reserved          ', decode('|'))
    print(decode('k--------------------------------------------------------------------------f'))
    print(decode('|'), '                                                                        ', decode('|'))
    print(decode('|'), '   1. Create the sample code for studying Deep Learning                 ', decode('|'))
    print(decode('|'), '              to be continue...                                         ', decode('|'))
    print(decode('p--------------------------------------------------------------------------q'))

def decode(x):
    return (''.join(dicArray.get(i,i.encode('utf-8')).decode('utf-8') for i in x))

def main():
    license_alarm()
    #print(serial_number_plus(),decode('------------------------------------------------------------------------'))
    
import os
import sys
for dirname, _, filenames in os.walk('.\data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
from keras.models import Sequential   #import Sequential model
from keras.layers import Dense        #import Dense loyer

df = pd.read_csv(".\data/train.csv")
dataset = df.values

train_feature_data = dataset[:,np.r_[2,4:8]]  #feature: cloumn_2, column_4, column_5, column_6,, column_7
train_target_data = dataset[:,1] 

df = pd.read_csv(".\data/test.csv")
dataset = df.values

test_feature_data = dataset[:,np.r_[1,3:7]]

df = pd.read_csv(".\data/gender_submission.csv")
dataset = df.values

test_target_data = dataset[:,1]


model = Sequential()
model.add(Dense(8, input_shape=(8,), activation="relu"))
model.add(Dense(8, activation='relu'))                      #hidden layer
model.add(1, activation="sigmoid")             #output layer 

model.summary()

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(train_feature_data,train_target_data, epochs=10, batch_size=10, verbose=0)

loss, accuracy = model.evaluate(train_feature_data, train_target_data)
print("訓練資料集的準確度 = {:.2f}".format(accuracy))
loss, accuracy = model.evaluate(test_feature_data, test_target_data)
print("測試資料集的準確度 = {:.2f}".format(accuracy))
# 測試資料集的預測值
Y_pred = model.predict_classes(test_feature_data, batch_size=10, verbose=0)
print(Y_pred[0], Y_pred[1]) 


# train_label_data= dataset[:,9:10]
# print(train_data)
# print(train_label_data)
# train_data = pd.read_csv(".\data/train.csv")
# dataset = train_data.values
# print(type(dataset))



# print(train_data+train_1_data)
# train_data.head()


# test_data = pd.read_csv('.\data/test.csv')
# test_data.head()

# women = train_data.loc[train_data.Sex == 'female']["Survived"]
# rate_women = sum(women)/len(women)

# print("% of women who survived:", rate_women)

# men = train_data.loc[train_data.Sex == 'male']["Survived"]
# rate_men = sum(men)/len(men)
# print("% of men who survived:", rate_men)



# from sklearn.ensemble import RandomForestClassifier

# y = train_data["Survived"]

# features = ["Pclass", "Sex", "SibSp", "Parch"]
# X = pd.get_dummies(train_data[features])
# X_test = pd.get_dummies(test_data[features])

# model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
# model.fit(X, y)
# predictions = model.predict(X_test)

# output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
# output.to_csv('submission.csv', index=False)
# print("Your submission was successfully saved!")


if __name__ == '__main__':
    main()