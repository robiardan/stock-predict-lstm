import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
import datetime
import tensorflow as tf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
import plotly.graph_objs as go

# ==============================
# Streamlit Title
# ==============================
st.title('📈 Stock Price Prediction with LSTM')

# Sidebar for date input
st.sidebar.subheader("📅 Select Date Range")
start_date = st.sidebar.date_input("Start Date", datetime.date(2010, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

if start_date >= end_date:
    st.error("❌ End date must be after start date.")
    st.stop()

# ==============================
# User Input for Stock
# ==============================
user_input = st.text_input('💡 Enter Stock Ticker Symbol', 'AAPL').upper()
start = start_date.strftime('%Y-%m-%d')
end = end_date.strftime('%Y-%m-%d')

# ==============================
# Fetch Data from Yahoo Finance
# ==============================
df = yf.download(user_input, start=start, end=end)
if df.empty:
    st.warning(f"No data found for {user_input} between {start} and {end}")
    st.stop()

# ==============================
# Data Summary
# ==============================
st.subheader('📊 Data Summary')
st.write(df.describe())

# ==============================
# Time Series Visualization
# ==============================
# Tampilkan judul
st.subheader(f'📉 {user_input} Close Price Over Time')

# Plot menggunakan matplotlib dan tampilkan di Streamlit
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df.index, df['Close'], label='Close Price', color='blue')
ax.set_title(f'{user_input} Close Price Time Series')
ax.set_xlabel('Date')
ax.set_ylabel('Close Price')
ax.grid(True)
plt.tight_layout()

st.pyplot(fig)

# ==============================
# Preprocessing: Normalization
# ==============================
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['Close']])

# Split data
train_size = int(len(scaled_data) * 0.80)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

def create_dataset(dataset, time_step=100):
    x, y = [], []
    for i in range(time_step, len(dataset)):
        x.append(dataset[i-time_step:i, 0])
        y.append(dataset[i, 0])
    return np.array(x), np.array(y)

time_step = 100
x_train, y_train = create_dataset(train_data, time_step)
x_test, y_test = create_dataset(test_data, time_step)

# Reshape
x_train = x_train.reshape(-1, time_step, 1)
x_test = x_test.reshape(-1, time_step, 1)

# ==============================
# Load Trained LSTM Model
# ==============================
try:
    best_model = load_model('keras_model.h5', custom_objects={'DTypePolicy': tf.keras.mixed_precision.Policy})
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# ==============================
# Predict & Evaluate
# ==============================
y_pred = best_model.predict(x_test).flatten()
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_pred_actual = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

mape_score = mean_absolute_percentage_error(y_test_actual, y_pred_actual)
st.success(f"✅ MAPE on Test Set: {mape_score * 100:.2f}%")

# ==============================
# Plot Predictions vs Actual
# ==============================
st.subheader('🔍 Prediction vs Actual')
fig2 = go.Figure()
fig2.add_trace(go.Scatter(y=y_test_actual, mode='lines', name='Actual', line=dict(color='blue')))
fig2.add_trace(go.Scatter(y=y_pred_actual, mode='lines', name='Predicted', line=dict(color='red')))
fig2.update_layout(title='LSTM Prediction vs Actual',
                   xaxis_title='Time Steps',
                   yaxis_title='Price',
                   template='plotly_white')
st.plotly_chart(fig2, use_container_width=True)

# ==============================
# Forecast Future Periods
# ==============================
forecast_periods = st.slider('🔮 Predict future periods:', min_value=1, max_value=24, value=6)

last_sequence = x_test[-1]
future_predictions = []

for _ in range(forecast_periods):
    next_pred = best_model.predict(last_sequence.reshape(1, time_step, 1)).flatten()[0]
    future_predictions.append(next_pred)
    last_sequence = np.append(last_sequence[1:], next_pred)

future_prices = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()

# ==============================
# Plot Forecast
# ==============================
st.subheader('📈 Forecasted Future Prices')
fig3 = go.Figure()
fig3.add_trace(go.Scatter(y=y_test_actual, mode='lines', name='Actual', line=dict(color='blue')))
fig3.add_trace(go.Scatter(y=y_pred_actual, mode='lines', name='Predicted', line=dict(color='orange')))
fig3.add_trace(go.Scatter(x=np.arange(len(y_test_actual), len(y_test_actual) + forecast_periods),
                          y=future_prices, mode='lines', name='Forecasted', line=dict(color='green')))
fig3.update_layout(title='Actual, Predicted, and Forecasted Prices',
                   xaxis_title='Time Steps',
                   yaxis_title='Stock Price',
                   template='plotly_white')
st.plotly_chart(fig3, use_container_width=True)

# ==============================
# Display Forecast Values
# ==============================
st.write(f"📅 Forecasted stock prices for the next {forecast_periods} periods:")
st.write(future_prices)
