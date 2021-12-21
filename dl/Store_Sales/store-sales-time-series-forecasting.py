
# refer https://www.kaggle.com/c/titanic
# from keras.activations import deserialize
from datetime import date
from typing import Optional
import numpy as np # linear algebra
import pandas as pd
from scipy.sparse import data

serial_number  = 0
DataPath="./kaggle/input/store-sales-time-series-forecasting/"
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
    print(decode('|'), '              Store-Sales-time-series-forecasting  - Practice           ', decode('|'))
    print(decode('|'), ' refer https://www.kaggle.com/c/store-sales-time-series-forecasting     ', decode('|'))
    print(decode('|'), '                              v0.03                                     ', decode('|'))
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
for dirname, _, filenames in os.walk('./kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# from keras.models import Sequential           #import Sequential model
# from keras.layers import Dense,Dropout        #import Dense loyer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.layers import Conv2D, MaxPool2D
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import RMSprop
from tensorflow.python.keras.layers.core import Dropout 

#EDA: Exploratory Data Analysis
def data_eda(src_df):
    print(src_df.head(5))
    print(decode('------')+"Check none (NULL)"+decode('------'))
    print(src_df.isnull().sum())
    print(decode('------')+"Check NA or NAN (Not Available, Not A Number)"+decode('------'))
    print(src_df.isna().sum())


# train_df = pd.read_csv(DataPath+"train.csv", index_col=0)
train_df = pd.read_csv(DataPath+"train.csv")
train_data = train_df.copy()
print(decode('------')+"train_df"+decode('------'))
# data_eda(train_df)
# test_df = pd.read_csv(DataPath+"test.csv", index_col=0)
test_df = pd.read_csv(DataPath+"test.csv")
test_data = test_df.copy()
print(decode('------')+"test_df"+decode('------'))
# data_eda(test_df)
oil_df = pd.read_csv(DataPath+"oil.csv")
print(decode('------')+"oil_df"+decode('------'))
# data_eda(oil_df)
holiday_event_df = pd.read_csv(DataPath+"holidays_events.csv")
print(decode('------')+"holiday_event_df"+decode('------'))
# data_eda(holiday_event_df)
transaction_df = pd.read_csv(DataPath+"transactions.csv")
print(decode('------')+"transaction_df"+decode('------'))
# data_eda(transaction_df)
store_df = pd.read_csv(DataPath+"stores.csv")
print(decode('------')+"store_df"+decode('------'))
# data_eda(store_df)

print(decode('------')+"combine data"+decode('------'))
#combine data (holidays_events/oil) base on date
pre_data = train_df.merge(holiday_event_df, on = 'date' ,how='left')
pre_data = pre_data.merge(oil_df, on = 'date' ,how='left')
#combine data (store_df) base on store_nbr
pre_data = pre_data.merge(store_df, on = 'store_nbr' ,how='left')
#combine data (transaction_df) base on date,store_nbr
pre_data = pre_data.merge(transaction_df, on = ['date','store_nbr'] ,how='left')

#Data modify
#rename (type_x base on "holidays_events.csv")
#       (type_y base on "stores.csv")
pre_data =pre_data.rename(columns={"type_x" : "event_type", "type_y" : "store_type"})

print("=========holiday data merge================")
train_data = train_data.merge(holiday_event_df, on = 'date' ,how='left')
test_data = test_data.merge(holiday_event_df, on = 'date' ,how='left')

train_data = train_data.rename(columns={"type" : "event_type"})
test_data  = test_data.rename(columns={"type" : "event_type"})
train_data["event_type"] = train_data.event_type.apply(lambda x : 6 if x=='Additional' else (
                                                                  5 if x=='Bridge' else (
                                                                  4 if x=='Event' else (
                                                                  3 if x=='Holiday' else (
                                                                  2 if x=='Transfer' else (
                                                                  1 if x=='Work Day' else 0))))))
test_data["event_type"] = test_data.event_type.apply(lambda x : 6 if x=='Additional' else (
                                                                  5 if x=='Bridge' else (
                                                                  4 if x=='Event' else (
                                                                  3 if x=='Holiday' else (
                                                                  2 if x=='Transfer' else (
                                                                  1 if x=='Work Day' else 0))))))
train_data["transferred"] = train_data.transferred.apply(lambda x: 1 if x=='TRUE' else 0)
test_data["transferred"] = test_data.transferred.apply(lambda x: 1 if x=='TRUE' else 0)

train_data["locale"] = train_data.locale.apply(lambda x : 3 if x=='National' else (
                                                                  2 if x=='Regional' else (
                                                                  1 if x=='Local' else 0 )))
test_data["locale"] = test_data.locale.apply(lambda x : 3 if x=='National' else (
                                                                  2 if x=='Regional' else (
                                                                  1 if x=='Local' else 0)))

train_data = train_data.drop(columns={'locale_name','description'})
test_data  = test_data.drop(columns={'locale_name','description'})



from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

#check Categorical variable
object_cols = [cname for cname in train_data.columns 
               if train_data[cname].dtype == "object" 
               and cname != "date"]
print("Categorical variables:")
print(object_cols) 
# train_data[object_cols].unique()    #item name
print(train_data[object_cols].nunique())  #number 

oen = OrdinalEncoder()
train_data[object_cols] = oen.fit_transform(train_data[object_cols])
oen = OrdinalEncoder()
test_data[object_cols]  = oen.fit_transform(test_data[object_cols])

print("=========oil data merge================")
train_data = train_data.merge(oil_df, on = 'date' ,how='left')
train_data["dcoilwtico"] = train_data.dcoilwtico.fillna(0)     #fill with NaN to zero
test_data = test_data.merge(oil_df, on = 'date' ,how='left')
test_data["dcoilwtico"] = test_data.dcoilwtico.fillna(0)     #fill with NaN to zero

print("=========transaction data merge================")
train_data = train_data.merge(transaction_df, on = ['date','store_nbr'] ,how='left')
test_data  = test_data.merge(transaction_df, on = ['date','store_nbr'] ,how='left')
train_data["transactions"] = train_data.transactions.fillna(0)     #fill with NaN to zero
test_data["transactions"] = test_data.transactions.fillna(0)       #fill with NaN to zero

#change date data
pre_data['date'] = pd.to_datetime(pre_data['date'])
pre_data['year'] = pre_data['date'].dt.year
pre_data['month'] = pre_data['date'].dt.month
pre_data['week'] = pre_data['date'].dt.isocalendar().week
pre_data['quarter'] = pre_data['date'].dt.quarter
pre_data['day_of_week'] = pre_data['date'].dt.day_name()

print("=========date data merge================")
train_data['date']      = pd.to_datetime(train_data['date'])
# train_data['year']        = train_data['date'].dt.year
train_data['month']       = train_data['date'].dt.month
train_data['quarter']     = train_data['date'].dt.quarter
train_data['day_of_week'] = train_data['date'].dt.dayofweek
# train_data['day_of_week'] = train_data['date'].dt.day_name()

test_data['date']      = pd.to_datetime(test_data['date'])
# test_data['year']        = test_data['date'].dt.year
test_data['month']       = test_data['date'].dt.month
test_data['quarter']     = test_data['date'].dt.quarter
test_data['day_of_week'] = test_data['date'].dt.dayofweek

print("=========stores data merge================")
# train_data.insert(loc=7, column='type', value=store_df['type'])  #not coresponse related row data.
#combine data (store_df) base on store_nbr
train_data = train_data.merge(store_df, on = 'store_nbr' ,how='left')
train_data  = train_data.rename(columns={"type" : "store_type"})
train_data["store_type"] = train_data.store_type.apply(lambda x : 5 if x=='A' else (
                                                                  4 if x=='B' else (
                                                                  3 if x=='C' else (
                                                                  2 if x=='D' else (
                                                                  1 if x=='E' else 0)))))
test_data  = test_data.merge(store_df, on = 'store_nbr' ,how='left')
test_data  = test_data.rename(columns={"type" : "store_type"})
test_data["store_type"] = test_data.store_type.apply(lambda x : 5 if x=='A' else (
                                                                4 if x=='B' else (
                                                                3 if x=='C' else (
                                                                2 if x=='D' else (
                                                                1 if x=='E' else 0)))))

train_data = train_data.drop(columns={'city','state'})
test_data  = test_data.drop(columns={'city','state'})

#fill with 0
# pre_data["transactions"] = pre_data.transactions.apply(lambda x: 0 if x=='female' else 0)
pre_data["transactions"] = pre_data.transactions.fillna(0)     #fill with NaN to zero
# data_eda(pre_data)


#maxmin normalization
print(decode('------')+"maxmin normalization"+decode('------'))
transaction_minmax = MinMaxScaler()
dcoilwtico_minmax = MinMaxScaler()
train_data['transactions'] = transaction_minmax.fit_transform(np.array(train_data['transactions']).reshape(-1,1))
train_data['dcoilwtico'] = dcoilwtico_minmax.fit_transform(np.array(train_data['dcoilwtico']).reshape(-1,1))
test_data['transactions'] = transaction_minmax.fit_transform(np.array(test_data['transactions']).reshape(-1,1))
test_data['dcoilwtico'] = dcoilwtico_minmax.fit_transform(np.array(test_data['dcoilwtico']).reshape(-1,1))

#group by 'store_type' and get mean.
storetype_sales_df = pre_data.groupby('store_type').agg({"sales" : "mean"}).reset_index().sort_values(by='sales', ascending=False)
# storetype_sales_df.to_csv("store_sale.csv")

#group by 'family' and get mean.
family_sales_df = pre_data.groupby('family').agg({"sales" : "mean"}).reset_index().sort_values(by='sales', ascending=False)[:10]
# family_sales_df.to_csv("family_sales.csv")

#group by 'cluster' and get mean
cluster_sales_df = pre_data.groupby('cluster').agg({"sales" : "mean"}).reset_index()
# cluster_sales_df.to_csv("cluster_sales.csv")

from plotly.subplots import make_subplots
import plotly.graph_objects as go

def show_eda_chart():
    #color 
    family_sales_df['color'] = '#008000'
    family_sales_df['color'][:3] = '#00FF00'
    cluster_sales_df['color'] = '#00FF00'
    
    #chart diagram
    figures = make_subplots(rows=2, cols=2, specs=[[{"type" : "bar"}, {"type" : "pie"}],[{"colspan" : 2}, None]],
     column_widths=[0.7, 0.3], vertical_spacing=0, horizontal_spacing=0.02,
      subplot_titles=("Top 10 highest Product Sales", "Highest Sales in Stores", "Clusters Vs Sales"))
    
    figures.add_trace(go.Bar(x=family_sales_df['sales'], y =family_sales_df['family'],
                    marker=dict(color=family_sales_df['color']), name='Family', orientation='h'),
                    row=1, col=1)
    # pre_data.to_csv("testwayne.csv")
    figures.add_trace(go.Pie(values=storetype_sales_df['sales'], labels=storetype_sales_df['store_type'], name='Store type',
                    marker=dict(colors=['#006400', '#008000','#228B22','#00FF00','#7CFC00','#00FF7F']), hole=0.7,
                    hoverinfo='label+percent+value', textinfo='label'), 
                    row=1, col=2)
    figures.add_trace(go.Bar(x=cluster_sales_df['cluster'], y=cluster_sales_df['sales'], 
                         marker=dict(color= cluster_sales_df['color']), name='Cluster'), 
                         row=2, col=1)
    
    figures.show()

# show_eda_chart()

import matplotlib.pyplot as plt

def show_oil_plt(oil_data):
    plt.plot(oil_data.set_index('date').dcoilwtico, color='green', label=f"oil_price")
    plt.title("Oil price vs Days")
    plt.xlabel("Days")
    plt.ylabel("Oil Price")
    plt.legend()
    plt.show()

# show_oil_plt(pre_data)

#date sales (per day) modify
date_sales_per_day_df = pre_data.groupby('date').agg({'sales':'mean'}).reset_index()
date_sales_per_day_df['weekly_avg_sales'] = date_sales_per_day_df['sales'].ewm(span=7, adjust=False).mean()
# date_sales_per_day_df.to_csv("date_sale_per_day.csv")
# date_sales_per_day_df.head()

#transaction (per day)
transaction_per_day_df = pre_data.groupby('date').agg({'transactions':'mean'}).reset_index()
transaction_per_day_df['weekly_avg_transactions'] = transaction_per_day_df['transactions'].ewm(span=7,adjust=False).mean()
# transaction_per_day_df.to_csv("transaction_per_day.csv")
transaction_per_day_df.head()

def show_sales_oil_plt():
    figures=make_subplots()

    figures.add_trace(go.Scatter(x=date_sales_per_day_df.date,y=date_sales_per_day_df.sales,name="Sales"))
    figures.add_trace(go.Scatter(x=date_sales_per_day_df.date,y=date_sales_per_day_df.weekly_avg_sales,name="Weekly Sales"))
    
    
    figures.add_trace(go.Scatter(x=oil_df.date,y=oil_df.dcoilwtico,name="Oil Price"))
    
    figures.update_layout(autosize=True,width=900,height=500,title_text="Variation of Sales and Oil Price Through Time")
    figures.update_xaxes(title_text="Days")
    figures.update_yaxes(title_text="Prices")
    figures.show()

# show_sales_oil_plt()

def show_transaction_oil_plt():
    figures=make_subplots()
    
    figures.add_trace(go.Scatter(x=transaction_per_day_df.date,y=transaction_per_day_df.transactions,name="Transactions"))
    figures.add_trace(go.Scatter(x=transaction_per_day_df.date,y=transaction_per_day_df.weekly_avg_transactions,name="Weekly Transactions"))
    
    figures.add_trace(go.Scatter(x=oil_df.date,y=oil_df.dcoilwtico,name="Oil Price"))
    
    figures.update_layout(autosize=True,width=900,height=500,title_text="Variation Transactions and Oil Price Through Time")
    figures.update_xaxes(title_text="Days")
    figures.update_yaxes(title_text="Prices")
    figures.show()

# show_transaction_oil_plt()


import seaborn as sns

#pearson correlation  (sales, transaction, oil)
def pearson_correlation_4_sale(modify_sales_df):
    modify_sales_df = modify_sales_df.drop(columns='weekly_avg_sales')
    modify_sales_df['sales'] = date_sales_per_day_df['sales']
    modify_sales_df['transactions'] = transaction_per_day_df['transactions']
    modify_sales_df['dcoilwtico'] = oil_df['dcoilwtico']
    plt.figure(figsize=(5,3))
    sns.heatmap(modify_sales_df.corr(), annot=True)

    plt.show()

# pearson_correlation_4_sale(date_sales_per_day_df)


from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

object_cols = [cname for cname in train_df.columns 
               if train_df[cname].dtype == "object" 
               and cname != "date"]

print("Categorical variables:")
print(object_cols) 

oen = OrdinalEncoder()
train_df[object_cols] = oen.fit_transform(train_df[object_cols])



#==================================================================================
print(decode('--------------------------------------------------------------'))

# train_data = train_df.groupby(['date']).agg({'sales':'mean', 'onpromotion':'mean'})
# print(train_data.tail())
# train_data.to_csv("check_data.csv")

train_dataset = train_data.values
x_train = train_dataset[0:, np.r_[2:4, 5:16]]
y_train = train_dataset[0:, 4]
x_train = np.asarray(x_train).astype('float32')
y_train = np.asarray(x_train).astype('float32')

test_dataset = test_data.values
x_test = test_dataset[0:, np.r_[2:15]]
x_test = np.asarray(x_train).astype('float32')
# y_test = test_dataset[0:, 4]


print(decode('--------------------------------------------------------------'))
num_feature_input = x_train.shape[1]
history_input = 30



from keras.preprocessing.sequence import TimeseriesGenerator
generator = TimeseriesGenerator(x_train, y_train, length=history_input, batch_size = 1)


from keras.layers import LSTM

def Multi_Step_LSTM_model():
    
    # Use Keras sequential model
    model = Sequential()    
    
    # First LSTM layer with Dropout regularisation; Set return_sequences to True to feed outputs to next layer
    model.add(LSTM(units = 30, activation='relu', return_sequences = True, input_shape = (history_input, num_feature_input))) 
    model.add(Dropout(0.2))
    
    # # Second LSTM layer with Dropout regularisation; Set return_sequences to True to feed outputs to next layer
    # model.add(LSTM(units = 50,  activation='relu', return_sequences = True))                                    
    # model.add(Dropout(0.2))
    
    # Final LSTM layer with Dropout regularisation; Set return_sequences to False since now we will be predicting with the output layer
    model.add(LSTM(units = 20))
    model.add(Dropout(0.2))
    
    # The output layer with linear activation to predict Open stock price
    model.add(Dense(units=1, activation = "linear"))
    
    return model



model = Multi_Step_LSTM_model()
model.summary()

model.compile(optimizer='adam', loss='mean_squared_error', metrics = ['accuracy'])

model.fit(generator, steps_per_epoch=len(generator), epochs=20, verbose=2)

full_dataset = pd.concat([train_data, test_df], ignore_index=True, sort=False)
full_dataset.to_csv("checkTest.csv")
full_dataset = full_dataset.iloc[3000887-5:,:]



full_dataset = full_dataset.groupby(['date']).agg({'sales':'mean', 'onpromotion':'mean'})
full_dataset.to_csv("checkTest12.csv")

