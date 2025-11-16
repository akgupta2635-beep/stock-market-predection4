import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load model
model = load_model('Stock Prediction Model.keras')

st.header('Stock Market Predictor')

# Input
stock = st.text_input('Enter Stock Symbol', 'GOOG')
start = '2012-01-01'
end = '2022-12-31'

# Download stock data
data = yf.download(stock, start, end)

st.subheader('Stock Data')
st.write(data)

# Train/Test Split
data_train = pd.DataFrame(data.Close[0 : int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80) : ])

# Scaling
scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(data_train)                    # ❗ Correct: Fit only on training data

past_100_days = data_train.tail(100)
data_test_all = pd.concat([past_100_days, data_test], ignore_index=True)

data_test_scaled = scaler.transform(data_test_all)   # ❗ Correct: Only transform

# Moving Averages
st.subheader('Price vs MA50')
ma_50 = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50, 'r')
plt.plot(data.Close, 'g')
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100 = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50, 'r')
plt.plot(ma_100, 'b')
plt.plot(data.Close, 'g')
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
ma_200 = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(ma_100, 'r')
plt.plot(ma_200, 'b')
plt.plot(data.Close, 'g')
st.pyplot(fig3)

# Prepare test data for prediction
x_test = []
y_test = []

for i in range(100, data_test_scaled.shape[0]):
    x_test.append(data_test_scaled[i-100:i])
    y_test.append(data_test_scaled[i, 0])

x_test = np.array(x_test)
y_test = np.array(y_test)

# Predict
predictions = model.predict(x_test)

# Inverse scaling to original price
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test.reshape(-1,1))

# Plot prediction vs original
st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(10,6))
plt.plot(y_test, 'g', label='Original Price')
plt.plot(predictions, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig4)
