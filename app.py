import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

st.title("📈 Stock Market Predictor")

model = load_model("StockMarket.keras")

stock = st.text_input("Enter Stock Symbol", "GOOG")
start = "2012-01-01"
end = "2022-12-31"

if st.button("Predict"):
    data = yf.download(stock, start=start, end=end)

    if data.empty:
        st.error("Invalid stock symbol or no data found!")
    else:
        st.success("Data loaded successfully!")
        st.write(data)

        close = data[['Close']]

        scaler = MinMaxScaler(feature_range=(0,1))
        scaled = scaler.fit_transform(close)

        lookback = 100
        last_100 = scaled[-lookback:]
        X_input = np.array(last_100).reshape(1, lookback, 1)

        pred = model.predict(X_input)
        predicted_price = scaler.inverse_transform(pred)[0][0]

        st.subheader("📌 Predicted Next Close Price:")
        st.write(predicted_price)

        fig = plt.figure(figsize=(10,5))
        plt.plot(close, label="Actual Close Price")
        plt.title(f"{stock} Close Price Chart")
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        plt.legend()

        st.pyplot(fig)
