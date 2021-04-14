#!/usr/bin/env python
# coding: utf-8

# In[78]:


## Algo trading homework 1 
### Saltanat Khalyk
## 201553895


# In[2]:


### LSTM


# In[3]:


import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


# In[4]:


BHP = web.DataReader('BHP', data_source='yahoo', start ='2012-01-01', end= '2021-03-14')


# In[5]:


BHP


# In[6]:


data = BHP.filter(['Close'])


# In[7]:


dataset = data.values
training_data_len = math.ceil(len(dataset)*.8)
training_data_len 


# In[8]:


# Scale the data
scaler = MinMaxScaler(feature_range =(0,1))
scaled_data = scaler.fit_transform(dataset)

scaled_data


# In[9]:


# create training dataset
# create the scaled training dataset
train_data =scaled_data[0:training_data_len, :]


x_train=[]
y_train =[]


for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i, 0])
    if i<=60:
        print(x_train)
        print(y_train)
        print()


# In[10]:


x_train, y_train = np.array(x_train), np.array(y_train)


# In[11]:


x_train.shape[0]


# In[17]:


#train_data[61-60:61,0]
#train_data[61,0]


# In[18]:


#x_train = np.reshape(x_train, (1788, 60,1))
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
#x_train.shape


# In[19]:


model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape = (x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(1))


# In[20]:


model.compile(optimizer='adam', loss='mean_squared_error')


# In[22]:


model.fit(x_train, y_train, batch_size=1, epochs=1)


# In[23]:


#create testing dataset
#create new array from index 1788 to 1848
test_data =scaled_data[training_data_len-60:, :]


x_test =[]
y_test =dataset[training_data_len:, :]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
    


# In[24]:


x_test =np.array(x_test)


# In[25]:


#LSTM expects 3d, number of features is 1, which is closing price
x_test =np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


# In[26]:


# get the models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)


# In[27]:


# get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean( predictions-y_test )**2 )
rmse


# In[29]:


#plot the data
train  = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions']= predictions
# visualize
plt.figure(figsize= (16,8))
plt.title('Model')
plt.xlabel('Date', fontsize =18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc ='lower right')
plt.show()


# In[30]:


valid


# In[32]:


Bhp =web.DataReader('BHP', data_source='yahoo', start='2012-01-01', end='2021-03-15')


# In[34]:


new_df = Bhp.filter(['Close'])


# In[35]:


last_60_days= new_df[-60:].values


# In[36]:


last_60_days_scaled = scaler.transform(last_60_days)


# In[37]:


### Prediction for 9th march closing price for BHP
X_test = []
X_test.append(last_60_days_scaled)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
pred_price = model.predict(X_test)
pred_price = scaler.inverse_transform(pred_price)
pred_price


# In[39]:


### BHP prediction price for 10th of march -- array([[75.46491]]
### because today is 09.03


# In[40]:


### DE company


# In[41]:


### Predicting for DE company 
DE =web.DataReader('DE', data_source='yahoo', start='2012-01-01', end='2021-03-15')
#DE = pd.read_csv('C:/Users/Asus/Documents/SALTANAT/fds/Homework01_algo_trading/Data/DE.csv')  


# In[42]:


#DE = DE[['Date', 'Close']]
new_df = DE.filter(['Close'])
last_60_days= new_df[-60:].values
last_60_days_scaled = scaler.transform(last_60_days)


# In[43]:


### Prediction for 10th march closing price for DE
X_test = []
X_test.append(last_60_days_scaled)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
pred_price = model.predict(X_test)
pred_price = scaler.inverse_transform(pred_price)
pred_price


# In[44]:


## DE - [[256.72595]]


# In[45]:


### FE company


# In[51]:


FE =web.DataReader('FE', data_source='yahoo', start='2012-01-01', end='2021-03-15')


# In[52]:


new_df = FE.filter(['Close'])
last_60_days= new_df[-60:].values
last_60_days_scaled = scaler.transform(last_60_days)


# In[53]:


X_test = []
X_test.append(last_60_days_scaled)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
pred_price = model.predict(X_test)
pred_price = scaler.inverse_transform(pred_price)
pred_price


# In[54]:


### FE price -- [[33.03623]]


# In[55]:


## GOOG company prediction


# In[56]:


GOOG =web.DataReader('GOOG', data_source='yahoo', start='2012-01-01', end='2021-03-15')


# In[57]:


new_df = GOOG.filter(['Close'])
last_60_days= new_df[-60:].values
last_60_days_scaled = scaler.transform(last_60_days)


# In[58]:


X_test = []
X_test.append(last_60_days_scaled)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
pred_price = model.predict(X_test)
pred_price = scaler.inverse_transform(pred_price)
pred_price


# In[59]:


### GOOG price [[308.8722]]


# In[61]:


## GS
GS =web.DataReader('GS', data_source='yahoo', start='2012-01-01', end='2021-03-15')


# In[62]:


new_df = GS.filter(['Close'])
last_60_days= new_df[-60:].values
last_60_days_scaled = scaler.transform(last_60_days)


# In[63]:


X_test = []
X_test.append(last_60_days_scaled)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
pred_price = model.predict(X_test)
pred_price = scaler.inverse_transform(pred_price)
pred_price


# In[64]:


### GS - [[249.32594]]


# In[66]:


### JNJ
JNJ =web.DataReader('JNJ', data_source='yahoo', start='2012-01-01', end='2021-03-15')
new_df = JNJ.filter(['Close'])
last_60_days= new_df[-60:].values
last_60_days_scaled = scaler.transform(last_60_days)
X_test = []
X_test.append(last_60_days_scaled)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
pred_price = model.predict(X_test)
pred_price = scaler.inverse_transform(pred_price)
pred_price


# In[67]:


## JNJ price [[150.28743]]


# In[69]:


### KO
KO =web.DataReader('KO', data_source='yahoo', start='2012-01-01', end='2021-03-15')
new_df = KO.filter(['Close'])
last_60_days= new_df[-60:].values
last_60_days_scaled = scaler.transform(last_60_days)
X_test = []
X_test.append(last_60_days_scaled)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
pred_price = model.predict(X_test)
pred_price = scaler.inverse_transform(pred_price)
pred_price


# In[70]:


## KO [[51.65556]]


# In[72]:


## T company
T =web.DataReader('T', data_source='yahoo', start='2012-01-01', end='2021-03-15')
new_df = T.filter(['Close'])
last_60_days= new_df[-60:].values
last_60_days_scaled = scaler.transform(last_60_days)
X_test = []
X_test.append(last_60_days_scaled)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
pred_price = model.predict(X_test)
pred_price = scaler.inverse_transform(pred_price)
pred_price


# In[73]:


## T company price prediction [[29.047464]]


# In[74]:


## WMT company
WMT =web.DataReader('WMT', data_source='yahoo', start='2012-01-01', end='2021-03-15')
new_df = WMT.filter(['Close'])
last_60_days= new_df[-60:].values
last_60_days_scaled = scaler.transform(last_60_days)
X_test = []
X_test.append(last_60_days_scaled)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
pred_price = model.predict(X_test)
pred_price = scaler.inverse_transform(pred_price)
pred_price


# In[75]:


## WMT -- [[125.75403]]


# In[76]:


XOM =web.DataReader('XOM', data_source='yahoo', start='2012-01-01', end='2021-03-15')
new_df = XOM.filter(['Close'])
last_60_days= new_df[-60:].values
last_60_days_scaled = scaler.transform(last_60_days)
X_test = []
X_test.append(last_60_days_scaled)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
pred_price = model.predict(X_test)
pred_price = scaler.inverse_transform(pred_price)
pred_price


# In[77]:


##  XOM [[61.71057]]


# In[ ]:




