import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
from keras.models import load_model
import streamlit as st
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
import plotly.graph_objs as go
import tensorflow as tf

start = '2010-01-01'
end = datetime.datetime.now().strftime('%Y-%m-%d')  # Mengambil tanggal saat ini sebagai string

st.title('Stock Price Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = yf.download(user_input, start=start, end=end)  # Menggunakan input pengguna untuk mengambil data

# Describing Data
st.subheader('Data from 2010-Now')
st.write(df.describe())

# Time Series Plot dengan Plotly
st.subheader('Stock Price Time Series')
fig = go.Figure()
fig.add_trace(go.Scatter(y=df['Close'], mode='lines', name='Stock Price', line=dict(color='blue')))
fig.update_layout(title='Stock Price Time Series', xaxis_title='Time', yaxis_title='Price')
st.plotly_chart(fig)

# Normalisasi data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['Close']])

# Splitting Data into train and test
train_data = scaled_data[0:int(len(scaled_data)*0.80)]
test_data = scaled_data[int(len(scaled_data)*0.80):]

# Mempersiapkan data dengan window size (timesteps)
def create_dataset(dataset, time_step=100):
    x, y = [], []
    for i in range(time_step, len(dataset)):
        x.append(dataset[i-time_step:i, 0])
        y.append(dataset[i, 0])
    return np.array(x), np.array(y)

time_step = 100
x_train, y_train = create_dataset(train_data, time_step)
x_test, y_test = create_dataset(test_data, time_step)

# Reshape data agar sesuai dengan input LSTM
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# Load model
try:
    best_model = tf.keras.models.load_model('keras_model.h5', custom_objects={'DTypePolicy': tf.keras.mixed_precision.Policy})
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Prediksi pada data pengujian
y_test_pred = best_model.predict(x_test).flatten()

# Denormalisasi data aktual dan prediksi
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_test_pred_denormalized = scaler.inverse_transform(y_test_pred.reshape(-1, 1)).flatten()

# Hitung MAPE untuk data pengujian setelah denormalisasi
test_mape_denormalized = mean_absolute_percentage_error(y_test_actual, y_test_pred_denormalized)
st.write(f"Test MAPE (Denormalized): {test_mape_denormalized*100:.2f}%")

# Plot LSTM Predictions vs Original dengan Plotly
st.subheader('LSTM Predictions vs Original')
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=np.arange(len(y_test_actual)), y=y_test_actual, mode='lines', name='Original Production', line=dict(color='blue')))
fig2.add_trace(go.Scatter(x=np.arange(len(y_test_pred_denormalized)), y=y_test_pred_denormalized, mode='lines', name='Predicted Production', line=dict(color='red')))
fig2.update_layout(title='LSTM Predictions vs Original Production',
                    xaxis_title='Time',
                    yaxis_title='Production')
st.plotly_chart(fig2)

# Forecasting beberapa periode ke depan
forecast_periods = st.slider('Select number of future periods to predict:', min_value=1, max_value=24, value=6, step=1)

# Ambil data terakhir dari data uji untuk memulai forecasting
last_sequence = x_test[-1]  # Data terakhir dari x_test
forecast_results = []

# Lakukan forecasting
for _ in range(forecast_periods):
    # Prediksi untuk periode berikutnya
    next_value = best_model.predict(last_sequence.reshape(1, time_step, 1)).flatten()[0]
    forecast_results.append(next_value)

    # Update last_sequence untuk prediksi berikutnya
    last_sequence = np.append(last_sequence[1:], next_value)

# Denormalisasi hasil forecast
forecast_results_denormalized = scaler.inverse_transform(np.array(forecast_results).reshape(-1, 1)).flatten()

# Plot hasil forecast dengan Plotly
fig3 = go.Figure()

# Actual values
fig3.add_trace(go.Scatter(
                x=np.arange(len(y_test_actual)),
                y=y_test_actual,
                mode='lines',
                name='Actual',
                line=dict(color='blue')
                ))

# Predicted values
fig3.add_trace(go.Scatter(
                x=np.arange(len(y_test_pred_denormalized)),
                y=y_test_pred_denormalized,
                mode='lines',
                name='Predicted',
                line=dict(color='orange')
                ))

# Forecasted values
fig3.add_trace(go.Scatter(
                x=np.arange(len(y_test_actual), len(y_test_actual) + forecast_periods),
                y=forecast_results_denormalized,
                mode='lines',
                name='Forecasted',
                line=dict(color='red')
                ))

fig3.update_layout(title='Actual, Predicted, and Forecasted Stock Prices',
                    xaxis_title='Time Steps',
                    yaxis_title='Stock Prices'
                )

st.plotly_chart(fig3)

# Menampilkan nilai hasil forecast
st.write(f"Forecasted values for the next {forecast_periods} periods:")
st.write(forecast_results_denormalized)
