import numpy as np
import pandas as pd
from matplotlib import pyplot
import matplotlib.pyplot as plt
from nsepy import get_history
from datetime import date
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

st.title("Stock Trend Prediction and Forecasting")

user_input = st.text_input("Enter Stock Ticker", "NIFTY 50")
symbol = [user_input]

df = []
df = pd.DataFrame(df)

for x in symbol:
    data = get_history(symbol=x, start=date(2012,3,1), end=date(2022,3,1), index = True)
    data = pd.DataFrame(data)
    data['Index_Name'] = x
    df = pd.concat([df,data])

    st.subheader("Data From 2012 till 2022")
    st.write(df)

    st.subheader("Features vs Time Chart")
    st.area_chart(df[["Open","High","Low","Close"]])


st.subheader("Closing Price vs Time Chart with 100MA")
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
plt.plot(ma100)
st.pyplot(fig)

st.subheader("Closing Price vs Time Chart with 100MA and 200MA")
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
plt.plot(ma100)
plt.plot(ma200)
st.pyplot(fig)

#Normalizing Dataset :
normalizer = MinMaxScaler(feature_range=(0,1))
data_scaled = normalizer.fit_transform(np.array(df.Close.values).reshape(-1,1))

train_size = int(len(data_scaled)*0.70)
test_size = len(data_scaled) - train_size
data_train, data_test = data_scaled[0:train_size,:], data_scaled[train_size:len(data_scaled),:1]

#Creating dataset in time series for LSTM model :
def create_data(dataset,step):
    Xtrain, Ytrain = [], []
    for i in range(len(dataset)-step-1):
        a = dataset[i:(i+step), 0]
        Xtrain.append(a)
        Ytrain.append(dataset[i + step, 0])
    return np.array(Xtrain), np.array(Ytrain)

#Taking 100 days price as one record for training :
time_stamp = 100
X_train, y_train = create_data(data_train,time_stamp)
X_test, y_test = create_data(data_test,time_stamp)

#Loading Model :
model = load_model("keras_model.h5")

#Predicitng on train and test data :
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

#Inverse transform to get actual value :
train_predict = normalizer.inverse_transform(train_predict)
test_predict = normalizer.inverse_transform(test_predict)

test = np.vstack((train_predict, test_predict))

st.subheader("Actual Closing Price vs Predicted Closing Price")
fig = plt.figure(figsize=(12,6))
plt.plot(normalizer.inverse_transform(data_scaled))
plt.plot(test)
st.pyplot(fig)

#Getting the last 100 days records :
fut_inp = data_test[int(len(data_test)-100):]
fut_inp = fut_inp.reshape(-1,1)

#Creating list of the last 100 data :
tmp_inp = list(fut_inp)
tmp_inp = tmp_inp[0].tolist()

#Predicting next 30 days price suing the current data :
#It will predict in sliding window manner (algorithm) with stride 1 :
lst_output = []
n_steps = 100
i = 0
while (i < 30):

    if (len(tmp_inp) > 100):
        fut_inp = np.array(tmp_inp[1:])
        fut_inp = fut_inp.reshape(-1, 1)
        fut_inp = fut_inp.reshape((1, n_steps, 1))
        yhat = model.predict(fut_inp, verbose=0)
        tmp_inp.extend(yhat[0].tolist())
        tmp_inp = tmp_inp[1:]
        lst_output.extend(yhat.tolist())
        i = i + 1
    else:
        fut_inp = fut_inp.reshape((1, n_steps, 1))
        yhat = model.predict(fut_inp, verbose=0)
        tmp_inp.extend(yhat[0].tolist())
        lst_output.extend(yhat.tolist())
        i = i + 1

#Creating final data for plotting :
data_new = data_scaled.tolist()
data_new.extend(lst_output)
final_graph = normalizer.inverse_transform(data_new).tolist()

#Plotting final results with predicted value after 30 Days :
fig = plt.figure(figsize=(12,6))
st.subheader("Prediction of Next Month Close")
plt.plot(final_graph,)
plt.ylabel("Price")
plt.xlabel("Time")
plt.title("{0} prediction of next month close".format(user_input))
plt.axhline(y=final_graph[len(final_graph)-1], color = 'red', linestyle = ':', label = 'NEXT 30D: {0}'.format(round(float(*final_graph[len(final_graph)-1]),2)))
plt.legend()
st.pyplot(fig)