import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import Callback

st.set_page_config(page_title="Stock Prediction App", layout="wide")
st.title("📈 Stock Price Prediction using ML & DL")
st.markdown("Enter a stock ticker (like **AAPL**, **TSLA**, or **BTC-USD**) and predict the next 30 days!")

# Input
ticker = st.text_input("Enter Stock Ticker Symbol:", "AAPL")
predict_button = st.button("🔮 Predict")

if predict_button:
    st.subheader(f"Fetching data for {ticker}...")
    data = yf.download(ticker, start="2020-01-01", end=pd.Timestamp.today())
    st.write("### Historical Data", data.tail())

    # Historical Closing Price Chart
    st.write("### 📊 Closing Price Chart")
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(data['Close'], label='Closing Price')
    ax.legend()
    st.pyplot(fig)

    # Data Preprocessing
    df = data[['Close']].values
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(df)
    training_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:training_size]
    test_data = scaled_data[training_size:]

    def create_dataset(dataset, time_step=60):
        X, y = [], []
        for i in range(len(dataset)-time_step-1):
            X.append(dataset[i:(i+time_step), 0])
            y.append(dataset[i + time_step, 0])
        return np.array(X), np.array(y)

    time_step = 60
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # --- LSTM Model Training with Streamlit progress ---
    st.write("### 🧠 Training LSTM Model...")
    training_status = st.empty()  # Placeholder for updates
    training_status.text("Training in progress... ⏳")

    class StreamlitProgress(Callback):
        def on_epoch_end(self, epoch, logs=None):
            training_status.text(f"Epoch {epoch+1}/5 completed, loss: {logs['loss']:.4f}")

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(60,1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=5, batch_size=32, callbacks=[StreamlitProgress()])

    training_status.text("✅ Training Completed!")

    # Forecast next 30 days
    last_60_days = scaled_data[-60:]
    temp_input = list(last_60_days.flatten())
    lst_output = []

    for i in range(30):
        x_input = np.array(temp_input[-60:])
        x_input = x_input.reshape(1,60,1)
        yhat = model.predict(x_input, verbose=0)
        temp_input.append(yhat[0][0])
        lst_output.append(yhat[0][0])

    forecast = scaler.inverse_transform(np.array(lst_output).reshape(-1,1))
    forecast_dates = pd.date_range(start=data.index[-1], periods=31, freq='D')[1:]
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Predicted Close': forecast.flatten()})

    st.write("### 📅 Predicted Prices for Next 30 Days")
    st.dataframe(forecast_df)

    # Plot forecast
    fig2, ax2 = plt.subplots(figsize=(6,3),dpi=80)
    ax2.plot(data['Close'], label='Historical')
    ax2.plot(forecast_df['Date'], forecast_df['Predicted Close'], label='Predicted', color='orange')
    ax2.legend()
    st.pyplot(fig2)

    st.success("✅ Prediction Completed!")
