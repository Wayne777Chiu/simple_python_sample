from datetime import date
from typing import Optional
import numpy as np # linear algebra
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.layers import Conv2D, MaxPool2D
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import RMSprop
from tensorflow.python.keras.layers.core import Dropout 
from scipy.sparse import data

DataPath="./kaggle/input/store-sales-time-series-forecasting/"
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

def decode(x):
    return (''.join(dicArray.get(i,i.encode('utf-8')).decode('utf-8') for i in x))

import os
for dirname, _, filenames in os.walk(DataPath):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train_df = pd.read_csv(DataPath+"train.csv")
train_data = train_df.copy()
print(decode('------')+"train_df")
# data_eda(train_df)
# test_df = pd.read_csv(DataPath+"test.csv", index_col=0)
test_df = pd.read_csv(DataPath+"test.csv")
test_data = test_df.copy()
print(decode('------')+"test_df")
# data_eda(test_df)
oil_df = pd.read_csv(DataPath+"oil.csv")
print(decode('------')+"oil_df")
# data_eda(oil_df)
holiday_event_df = pd.read_csv(DataPath+"holidays_events.csv")
print(decode('------')+"holiday_event_df")
# data_eda(holiday_event_df)
transaction_df = pd.read_csv(DataPath+"transactions.csv")
print(decode('------')+"transaction_df")
# data_eda(transaction_df)
store_df = pd.read_csv(DataPath+"stores.csv")
print(decode('------')+"store_df")
# data_eda(store_df)




print(decode('------')+"holiday data merge")
train_data = train_data.merge(holiday_event_df, on = 'date' ,how='left')
test_data = test_data.merge(holiday_event_df, on = 'date' ,how='left')

train_data = train_data.rename(columns={"type" : "event_type"})
test_data  = test_data.rename(columns={"type" : "event_type"})

print(decode('------')+"stores data merge")
# train_data.insert(loc=7, column='type', value=store_df['type'])  #not coresponse related row data.
#combine data (store_df) base on store_nbr
train_data = train_data.merge(store_df, on = 'store_nbr' ,how='left')
train_data  = train_data.rename(columns={"type" : "store_type"})
test_data  = test_data.merge(store_df, on = 'store_nbr' ,how='left')
test_data  = test_data.rename(columns={"type" : "store_type"})

print(decode('p-----')+"change store type to numberlize")
train_data["store_type"] = train_data.store_type.apply(lambda x : 5 if x=='A' else (
                                                                  4 if x=='B' else (
                                                                  3 if x=='C' else (
                                                                  2 if x=='D' else (
                                                                  1 if x=='E' else 0)))))
test_data["store_type"] = test_data.store_type.apply(lambda x : 5 if x=='A' else (
                                                                4 if x=='B' else (
                                                                3 if x=='C' else (
                                                                2 if x=='D' else (
                                                                1 if x=='E' else 0)))))

print(decode('------')+"transaction data merge")
train_data = train_data.merge(transaction_df, on = ['date','store_nbr'] ,how='left')
test_data  = test_data.merge(transaction_df, on = ['date','store_nbr'] ,how='left')
print(decode('p-----')+"fill zero for column:transactions ")
train_data["transactions"] = train_data.transactions.fillna(0)     #fill with NaN to zero
test_data["transactions"] = test_data.transactions.fillna(0)       #fill with NaN to zero

print(decode('------')+"oil data merge")
train_data = train_data.merge(oil_df, on = 'date' ,how='left')
test_data = test_data.merge(oil_df, on = 'date' ,how='left')
print(decode('p-----')+"fill zero for column:dcoilwtico ")
train_data["dcoilwtico"] = train_data.dcoilwtico.fillna(0)     #fill with NaN to zero
test_data["dcoilwtico"] = test_data.dcoilwtico.fillna(0)     #fill with NaN to zero

print(decode('------')+"mixed data merge")
print(decode('p-----')+"Add event place")
conditions = [
    ((train_data['locale']=='National') & (train_data['transferred']!=1)),
    ((train_data['locale']=='Regional') & (train_data['locale_name']==train_data['state'])),
    ((train_data['locale']=='Local') & (train_data['locale_name']==train_data['city']) & (train_data['transferred']!=1))]
choices = [1,1,1]   
train_data["get_event"] = np.select(conditions, choices, default=0)
conditions = [
    ((test_data['locale']=='National') & (test_data['transferred']!=1)),
    ((test_data['locale']=='Regional') & (test_data['locale_name']==test_data['state'])),
    ((test_data['locale']=='Local') & (test_data['locale_name']==test_data['city']) & (test_data['transferred']!=1))]
choices = [1,1,1]   
test_data["get_event"] = np.select(conditions, choices, default=0)

print(decode('------')+"date data merge")
# print(type(train_data['date']))
train_data['date']      = pd.to_datetime(train_data['date'])
train_data['year']        = train_data['date'].dt.year
train_data['month']       = train_data['date'].dt.month
train_data['quarter']     = train_data['date'].dt.quarter
train_data['day_of_week'] = train_data['date'].dt.dayofweek
# train_data['day_of_week'] = train_data['date'].dt.day_name()

test_data['date']      = pd.to_datetime(test_data['date'])
test_data['year']        = test_data['date'].dt.year
test_data['month']       = test_data['date'].dt.month
test_data['quarter']     = test_data['date'].dt.quarter
test_data['day_of_week'] = test_data['date'].dt.dayofweek

#maxmin normalization
print(decode('------')+"maxmin normalization")
from sklearn.preprocessing import MinMaxScaler
transaction_minmax = MinMaxScaler()
dcoilwtico_minmax = MinMaxScaler()
onpromotion_minmax = MinMaxScaler()
train_data['transactions'] = transaction_minmax.fit_transform(np.array(train_data['transactions']).reshape(-1,1))
train_data['dcoilwtico'] = dcoilwtico_minmax.fit_transform(np.array(train_data['dcoilwtico']).reshape(-1,1))
train_data['onpromotion'] = dcoilwtico_minmax.fit_transform(np.array(train_data['onpromotion']).reshape(-1,1))
test_data['transactions'] = transaction_minmax.fit_transform(np.array(test_data['transactions']).reshape(-1,1))
test_data['dcoilwtico'] = dcoilwtico_minmax.fit_transform(np.array(test_data['dcoilwtico']).reshape(-1,1))
test_data['onpromotion'] = dcoilwtico_minmax.fit_transform(np.array(test_data['onpromotion']).reshape(-1,1))

print(decode('------')+"drop redundancy columns")
train_data = train_data.drop(columns={'event_type', 'locale', 'locale_name', 'description', 'transferred','city','state'})
test_data  = test_data.drop(columns={'event_type', 'locale', 'locale_name', 'description', 'transferred','city','state'})

# family_group_df = train_data.groupby('family')
# family_group_df.i

from keras.preprocessing.sequence import TimeseriesGenerator
from keras.layers import LSTM
import matplotlib.pyplot as plt

prop = 0.8

def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train], label=train)
    plt.plot(train_history.history[validation], label=validation)
    plt.title("Train History")
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.show()


final_result = test_df.copy()
y_total_result=pd.DataFrame(columns={'id','sales'})
for group_name,df_group  in train_data.groupby('family'):
    X_train_df = train_data.loc[train_data['family']==group_name]
    X_test_df  = test_data.loc[test_data['family']==group_name]
    for group_name_2,df_group2  in X_train_df.groupby('store_nbr'):
        x_train_dataset = X_train_df.copy()
        x_train_dataset = x_train_dataset.loc[x_train_dataset['store_nbr']==group_name_2]
        x_test_dataset  = X_test_df.copy()
        x_test_dataset  = x_test_dataset.loc[x_test_dataset['store_nbr']==group_name_2]
        # tmp_text = group_name
        # if ('/' in tmp_text):
        #     tmp_text = tmp_text.replace("/","_") 
        # x_test_dataset.to_csv("./tmp/"+tmp_text+"_"+str(group_name_2)+"_test.csv")

        
        x_train_value = x_train_dataset.values
        x_train, x_validation = x_train_value[:round(len(x_train_value)*prop), np.r_[5,8:15]], x_train_value[round(len(x_train_value)*prop):, np.r_[5,8:15]]
        y_train, y_validation = x_train_value[:round(len(x_train_value)*prop), 4], x_train_value[round(len(x_train_value)*prop):, 4]
        
        x_train, x_validation = np.asarray(x_train).astype('float32'), np.asarray(x_validation).astype('float32')
        y_train, y_validation = np.asarray(y_train).astype('float32'), np.asarray(y_validation).astype('float32')

        x_validation = x_validation[:,np.newaxis]
        
        test_value = x_test_dataset.values
        x_test = test_value[0:, np.r_[4, 7:14]]
        x_test = np.asarray(x_test).astype('float32')
        x_test = x_test[:, np.newaxis]

        num_feature_input = x_train.shape[1]
        history_input = 1

        print(len(x_train))
        print("===")
        print(x_train.shape[1])
        generator = TimeseriesGenerator(x_train, y_train, length=history_input, batch_size = 1)
        for i in range(len(generator)):
            x, y = generator[i]
            print('%s => %s' % (x, y))
            break
        print(len(generator))

        def Multi_Step_LSTM_model():
            # # Use Keras sequential model
            model = Sequential()    
            # # First LSTM layer with Dropout regularisation; Set return_sequences to True to feed outputs to next layer
            model.add(LSTM(units=100, activation='relu', input_shape=(history_input,x_train.shape[1]), return_sequences=True)) 
            model.add(Dropout(0.2))
            # # # Second LSTM layer with Dropout regularisation; Set return_sequences to True to feed outputs to next layer
            model.add(LSTM(units=50,  activation='relu', return_sequences=True))                                    
            model.add(Dropout(0.2))
            model.add(Flatten())
            # # The output layer with linear activation to predict Open stock price
            model.add(Dense(units=1, activation = "linear"))
            return model

        print(decode('------')+"model construction")
        model = Multi_Step_LSTM_model()
        model.summary()
        model.compile(optimizer='adam', loss='mean_squared_error', metrics = ['mse'])
        train_history = model.fit(generator, steps_per_epoch=len(generator), epochs=50, verbose=2, validation_data=(x_validation, y_validation))

        # show_train_history(train_history, 'mse', 'val_mse')
        # show_train_history(train_history, 'loss', 'val_loss')

        result = model.predict(x_test)
        y_result=pd.DataFrame(columns={'id','sales'})
        y_result['id'] = x_test_dataset['id'].values
        y_result['sales'] = result
        y_total_result = pd.concat([y_total_result, y_result], ignore_index=True, axis=0)
        print(y_total_result)
        y_total_result.to_csv("y_total_result.csv")
          
          

final_result = final_result.merge(holiday_event_df, on = 'id' ,how='left')
output = pd.DataFrame({'id': final_result.id, 'sales': final_result.sales})
output.to_csv('submission.csv', index=False)
final_result.to_csv("final_result.csv")
print("Your submission was successfully saved!")
