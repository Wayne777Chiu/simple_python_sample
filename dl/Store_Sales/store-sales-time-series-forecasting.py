
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
    print(decode('|'), '                              v0.02                                     ', decode('|'))
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


train_df = pd.read_csv(DataPath+"train.csv", index_col=0)
print(decode('------')+"train_df"+decode('------'))
# data_eda(train_df)
test_df = pd.read_csv(DataPath+"test.csv", index_col=0)
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

#change date data
pre_data['date'] = pd.to_datetime(pre_data['date'])
pre_data['year'] = pre_data['date'].dt.year
pre_data['month'] = pre_data['date'].dt.month
pre_data['week'] = pre_data['date'].dt.isocalendar().week
pre_data['quarter'] = pre_data['date'].dt.quarter
pre_data['day_of_week'] = pre_data['date'].dt.day_name()

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

show_sales_oil_plt()

def show_transaction_oil_plt():
    figures=make_subplots()
    
    figures.add_trace(go.Scatter(x=transaction_per_day_df.date,y=transaction_per_day_df.transactions,name="Transactions"))
    figures.add_trace(go.Scatter(x=transaction_per_day_df.date,y=transaction_per_day_df.weekly_avg_transactions,name="Weekly Transactions"))
    
    figures.add_trace(go.Scatter(x=oil_df.date,y=oil_df.dcoilwtico,name="Oil Price"))
    
    figures.update_layout(autosize=True,width=900,height=500,title_text="Variation Transactions and Oil Price Through Time")
    figures.update_xaxes(title_text="Days")
    figures.update_yaxes(title_text="Prices")
    figures.show()

show_transaction_oil_plt()

import seaborn as sns

#pearson correlation  (sales, transaction, oil)
def pearson_correlation_4_oil():
    oil_df['sales'] = date_sales_per_day_df['sales']
    oil_df['transactions'] = transaction_per_day_df['transactions']
    plt.figure(figsize=(11,3))
    sns.heatmap(oil_df.corr(), annot=True)
    plt.show()

pearson_correlation_4_oil()



#==================================================================================


train_data = train_df.groupby(['date']).agg({'sales':'mean', 'onpromotion':'mean'})
train_data.tail()



x_train = train_data.copy()
y_train = train_data.sales.copy()

num_feature_input = len(x_train.columns)
history_input = 30

from keras.preprocessing.sequence import TimeseriesGenerator
generator = TimeseriesGenerator(x_train, y_train, length=history_input, batch_size = 1)


from keras.layers import LSTM

def Multi_Step_LSTM_model():
    
    # Use Keras sequential model
    model = Sequential()    
    
    # First LSTM layer with Dropout regularisation; Set return_sequences to True to feed outputs to next layer
    model.add(LSTM(units = 50, activation='relu', return_sequences = True, input_shape = (history_input, num_feature_input))) 
    model.add(Dropout(0.2))
    
    # Second LSTM layer with Dropout regularisation; Set return_sequences to True to feed outputs to next layer
    model.add(LSTM(units = 50,  activation='relu', return_sequences = True))                                    
    model.add(Dropout(0.2))
    
    # Final LSTM layer with Dropout regularisation; Set return_sequences to False since now we will be predicting with the output layer
    model.add(LSTM(units = 50))
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