
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
    print(decode('|'), '                              v0.05                                     ', decode('|'))
    print(decode('|'), '            Copyright (c) Wayne Chiu 2021. All Rights Reserved          ', decode('|'))
    print(decode('k--------------------------------------------------------------------------f'))
    print(decode('|'), '                                                                        ', decode('|'))
    print(decode('|'), '   1. Create the sample code for studying Deep Learning                 ', decode('|'))
    print(decode('|'), '              to be continue...                                         ', decode('|'))
    print(decode('p--------------------------------------------------------------------------q'))

def decode(x):
    return (''.join(dicArray.get(i,i.encode('utf-8')).decode('utf-8') for i in x))

def showplt():
    import matplotlib.pyplot as plt

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
        
# from keras.models import Sequential   #import Sequential model
# from keras.layers import Dense,Dropout        #import Dense loyer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten

#new column filter
new_column_names = ["_female", 
                  "_male",
                "_port_C", 
                "_port_Q",
                "_port_S",
                "_app_Dr",
                "_app_mr",
                "_app_master",
                "_app_miss",
                "_app_mrs"]

# df = pd.read_csv("./kaggle/input/titanic/train.csv").replace(regex={'female': 1, 'male': 0})   #set female as 1  and male as 0
df = pd.read_csv("./kaggle/input/titanic/train.csv")

#add new colunm to filter
df = df.reindex(columns = df.columns.tolist()+ new_column_names)

## Sex   total 2 column
df["_female"] = df.Sex.apply(lambda x: 1 if x=='female' else 0)
df["_male"]   = df.Sex.apply(lambda x: 1 if x=='male' else 0)

## title  DR./ Mr./ Master/ Miss./ Mrs.    total 5 column
df["_app_Dr"]     = df.apply(lambda x: 1 if "Dr." in x['Name'] else 0, axis=1)
df["_app_mr"]     = df.apply(lambda x: 1 if "Mr." in x['Name'] else 0, axis=1)
df["_app_master"] = df.apply(lambda x: 1 if "Master." in x['Name'] else 0, axis=1)
df["_app_miss"]   = df.apply(lambda x: 1 if "Miss." in x['Name'] else 0, axis=1)
df["_app_mrs"]    = df.apply(lambda x: 1 if "Mrs." in x['Name'] else 0, axis=1)
# print(df[df['Name'].astype(str).str.contains("Miss.")])

## Port of Embarkation: C = Cherbourg, Q = Queenstown, S = Southampton   total 3 column
df["_port_C"] = df.Embarked.apply(lambda x: 1 if x=='C' else 0)
df["_port_Q"] = df.Embarked.apply(lambda x: 1 if x=='Q' else 0)
df["_port_S"] = df.Embarked.apply(lambda x: 1 if x=='S' else 0)

df = df.fillna(0)     #fill with NaN to zero
dataset = df.values

#set random to noise.
np.random.shuffle(dataset)

# train data set # validation data set
#feature: Pclass(2), Age(5), SibSp(6), Parch(7), _female(12), _male(13), _port_C(14), _port_Q(15), _port_S(16)
#         _app_Dr(17), _app_mr(18), _app_master(19), _app_miss(20), _app_mrs(21)
train_feature_data, valid_feature_data = dataset[:891,np.r_[2,5:8, 12:22]], dataset[777:,np.r_[2,5:8, 12:22]]
# train_feature_data -= train_feature_data.mean(axis=0)
# train_feature_data /= 255.0

# valid_feature_data -= valid_feature_data.mean(axis=0)
# valid_feature_data /= 255.0
train_feature_data, valid_feature_data = np.asarray(train_feature_data).astype('float32'), np.asarray(valid_feature_data).astype('float32')
train_target_data, valid_target_data = dataset[:891,1], dataset[777:,1] 
train_target_data, valid_target_data = np.asarray(train_target_data).astype('float32'), np.asarray(valid_target_data).astype('float32')

# test data set
# test_data_df = pd.read_csv("./kaggle/input/titanic/test.csv").replace(regex={'female': 1, 'male': 0})  #set female as 1  and male as 0
test_data_df = pd.read_csv("./kaggle/input/titanic/test.csv")

#add new colunm to filter
test_data_df = test_data_df.reindex(columns = test_data_df.columns.tolist()+ new_column_names)

## Sex   total 2 column
test_data_df["_female"] = test_data_df.Sex.apply(lambda x: 1 if x=='female' else 0)
test_data_df["_male"]   = test_data_df.Sex.apply(lambda x: 1 if x=='male' else 0)

## title  DR./ Mr./ Master/ Miss./ Mrs.    total 5 column
test_data_df["_app_Dr"]     = test_data_df.apply(lambda x: 1 if "Dr." in x['Name'] else 0, axis=1)
test_data_df["_app_mr"]     = test_data_df.apply(lambda x: 1 if "Mr." in x['Name'] else 0, axis=1)
test_data_df["_app_master"] = test_data_df.apply(lambda x: 1 if "Master." in x['Name'] else 0, axis=1)
test_data_df["_app_miss"]   = test_data_df.apply(lambda x: 1 if "Miss." in x['Name'] else 0, axis=1)
test_data_df["_app_mrs"]    = test_data_df.apply(lambda x: 1 if "Mrs." in x['Name'] else 0, axis=1)
# print(df[df['Name'].astype(str).str.contains("Miss.")])

## Port of Embarkation: C = Cherbourg, Q = Queenstown, S = Southampton   total 3 column
test_data_df["_port_C"] = test_data_df.Embarked.apply(lambda x: 1 if x=='C' else 0)
test_data_df["_port_Q"] = test_data_df.Embarked.apply(lambda x: 1 if x=='Q' else 0)
test_data_df["_port_S"] = test_data_df.Embarked.apply(lambda x: 1 if x=='S' else 0)

test_data_df = test_data_df.fillna(0)   #fill with NaN to zero

dataset = test_data_df.values
#feature: Pclass(1), Age(4), SibSp(5), Parch(6), _female(11), _male(12), _port_C(13), _port_Q(14), _port_S(15)
#         _app_Dr(16), _app_mr(17), _app_master(18), _app_miss(19), _app_mrs(20)
test_feature_data = dataset[:,np.r_[1,4:7, 11:21]]   #feature: cloumn_1, column_3, column_4, column_5, column_6
test_feature_data = np.asarray(test_feature_data).astype('float32')

df = pd.read_csv("./kaggle/input/titanic/gender_submission.csv")
dataset = df.values
test_target_data = dataset[:,1]
test_target_data = np.asarray(test_target_data).astype('float32')


model = Sequential()
model.add(Flatten(input_shape=(14,)))                  #input  layer    5 cell input  
model.add(Dense(196, activation='relu'))               #hidden layer
model.add(Dense(38416, activation='relu'))              #hidden layer

model.add(Dense(1,activation="sigmoid"))              #output layer 

model.summary()

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# [print(i.shape, i.dtype) for i in model.inputs]
# [print(o.shape, o.dtype) for o in model.outputs]
# [print(l.name, l.input_shape, l.dtype) for l in model.layers]
# print("=====================================================")

history = model.fit(train_feature_data, train_target_data, epochs= 100, validation_data=(valid_feature_data, valid_target_data))

showplt()



# model.fit(train_feature_data,train_target_data, epochs=10, batch_size=10, verbose=0)

# loss, accuracy = model.evaluate(train_feature_data, train_target_data)
# print("Train data accuracy = {:.2f}".format(accuracy))
# loss, accuracy = model.evaluate(test_feature_data, test_target_data)
# print("Test data accuracy = {:.2f}".format(accuracy))

# Test data prediction
Y_pred = model.predict(test_feature_data)
# # print([i for item in Y_pred for i in item])
classes_Y_pred = np.around(Y_pred).astype(int)  
# print([i for item in classes_Y_pred for i in item])
loss, accuracy = model.evaluate(test_feature_data, test_target_data)
print("Test data accuracy = {:.2f}".format(accuracy))

output = pd.DataFrame({'PassengerId': test_data_df.PassengerId, 'Survived': [i for item in classes_Y_pred for i in item]})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")

# if __name__ == '__main__':
#     main()